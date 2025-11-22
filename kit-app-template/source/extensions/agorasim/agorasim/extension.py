# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
import omni.ext
import omni.ui as ui
import omni.usd
import omni.kit.app as kit_app
import omni.kit.commands

# --- Physics / USD ---
import omni.physx
from omni.physx import get_physx_scene_query_interface
import carb
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf
import numpy as np

# Optional extras
try:
    import warp as wp
    _HAS_WARP = True
except Exception as e:
    print(f"[AgoraSim] WARN: warp not available: {e}")
    _HAS_WARP = False

# Navigation
try:
    import omni.anim.navigation.core as nav_core
    _HAS_ISAAC_NAV = True
except Exception as e:
    print(f"[AgoraSim] WARN: Isaac navigation not available: {e}")
    _HAS_ISAAC_NAV = False

# --- BLACK & WHITE STYLE ---
VISIBLE_STYLE = {
    "Window": {
        "background_color": 0xFF_1A1A1A,
        "border_color": 0xFF_FFFFFF,
        "border_width": 1,
        "border_radius": 8,
    },
    "Header": {"color": 0xFF_FFFFFF, "font_size": 22, "alignment": ui.Alignment.CENTER},
    "Label": {"color": 0xFF_DDDDDD, "font_size": 14},
    "Button": {
        "color": 0xFF_FFFFFF,
        "border_color": 0xFF_FFFFFF,
        "border_width": 1,
        "background_color": 0x00_000000,
        "font_size": 14,
        "margin": 4,
        "padding": 4,
    },
    "Section": {
        "background_color": 0xFF_2A2A2A,
        "border_radius": 6,
        "padding": 6,
        "margin": 4,
    },
    "StatusGood": {"color": 0xFF_FFFFFF, "font_size": 14},
    "StatusBad": {"color": 0xFF_A0A0A0, "font_size": 14},
    "StatusError": {"color": 0xFF_FF5555, "font_size": 14},
}


class AgoraSimExtension(omni.ext.IExt):
    """Black & White AgoraSim Crowd Simulator (with PhysX corral + raycast gating)."""

    # ---------------- Warp kernel ----------------
    @staticmethod
    def _define_kernels():
        if not _HAS_WARP:
            return

        @wp.kernel
        def _move_agents(positions: wp.array(dtype=wp.vec2),
                         waypoints: wp.array(dtype=wp.vec2),
                         waypoint_indices: wp.array(dtype=wp.int32),
                         speed: float,
                         dt: float,
                         arrive_radius: float):
            """
            Move agents toward their current waypoint with arrival behavior.
            When close to waypoint, slow down smoothly.
            """
            tid = wp.tid()
            pos = positions[tid]
            waypoint_idx = waypoint_indices[tid]

            # If waypoint index is -1, agent has no path (don't move)
            if waypoint_idx < 0:
                return

            target = waypoints[waypoint_idx]
            d = target - pos
            eps = 1.0e-8
            dist = wp.length(d)

            if dist > eps:
                dir = d / dist

                # Arrival behavior: slow down near waypoint
                if dist < arrive_radius:
                    # Smoothly reduce speed as we approach
                    speed_factor = dist / arrive_radius
                    actual_speed = speed * speed_factor
                else:
                    actual_speed = speed

                # Move toward waypoint
                move_dist = actual_speed * dt
                if move_dist < dist:
                    positions[tid] = pos + dir * move_dist
                else:
                    # Reached waypoint exactly
                    positions[tid] = target

        AgoraSimExtension._move_agents = _move_agents

    # ---------------- Lifecycle ----------------
    def on_startup(self, ext_id):
        print("[AgoraSim] Black & White UI starting...")
        self._ext_id = ext_id
        self._is_running = False
        self._num_agents = 0
        self._agent_prims = []
        self._positions_wp = None
        self._speed = 6.0
        self._pi = None  # PointInstancer prim

        # Path following
        self._agent_paths = []  # List of paths, one per agent (each path is list of waypoints)
        self._agent_waypoint_idx = None  # Current waypoint index for each agent (Warp array)
        self._waypoints_wp = None  # All waypoints in one flat array
        self._waypoint_reach_distance = 0.5  # Distance to consider waypoint "reached"
        self._arrive_radius = 3.0  # Distance to start slowing down

        # FPS tracking
        self._fps_history = []
        self._fps_window_size = 60  # Average over 60 frames

        # NavMesh
        self._navmesh = None
        self._navmesh_volume = None
        self._navmesh_handle = None  # Actual navmesh object from nav_core
        self._init_navmesh_config()

        if _HAS_WARP:
            wp.init()
            self._define_kernels()
            # No longer using single target - agents will follow individual paths
        else:
            pass

        self._build_ui()

        # subscribe to app updates
        app = kit_app.get_app()
        self._update_sub = None
        try:
            evt_stream = None
            if hasattr(app, "get_update_event_stream"):
                evt_stream = app.get_update_event_stream()
            if evt_stream is not None:
                if hasattr(evt_stream, "create_subscription_to_pop"):
                    try:
                        self._update_sub = evt_stream.create_subscription_to_pop(self._on_update, name="AgoraSimUpdate")
                    except Exception:
                        self._update_sub = None
                if self._update_sub is None and hasattr(evt_stream, "create_subscription"):
                    try:
                        self._update_sub = evt_stream.create_subscription(self._on_update, name="AgoraSimUpdate")
                    except Exception:
                        self._update_sub = None
                if self._update_sub is None and hasattr(evt_stream, "subscribe"):
                    try:
                        self._update_sub = evt_stream.subscribe(self._on_update)
                    except Exception:
                        self._update_sub = None
            if self._update_sub is None:
                print("[AgoraSim] WARN: failed to create update subscription; no per-frame updates.")
        except Exception as e:
            print(f"[AgoraSim] WARN: exception while subscribing to update events: {e}")

        self._update_call_count = 0

        # Ensure PhysicsScene exists
        self._ensure_physics_scene()

        # Build initial agents
        self._reset_simulation(200)

        # auto-start convenience
        try:
            if _HAS_WARP and getattr(self, '_num_agents', 0) > 0:
                # Auto-generate initial paths (NavMesh if available, else idle)
                try:
                    self._generate_agent_paths()
                except Exception as e:
                    print(f"[AgoraSim] WARN: Failed to auto-generate paths: {e}")

                self._is_running = True
                try:
                    self._run_btn.text = "Stop"
                except Exception:
                    pass
                self._update_status("Running", "good")
            try:
                self._frame_agents()
            except Exception:
                pass
        except Exception:
            pass

    def on_shutdown(self):
        print("[AgoraSim] shutdown")
        if getattr(self, "_update_sub", None):
            try:
                sub = self._update_sub
                if hasattr(sub, "unsubscribe"):
                    sub.unsubscribe()
                elif hasattr(sub, "disconnect"):
                    sub.disconnect()
                elif callable(sub):
                    try:
                        sub()
                    except Exception:
                        pass
            except Exception as e:
                print(f"[AgoraSim] WARN: error unsubscribing: {e}")
        if getattr(self, "_window", None):
            self._window.destroy()

    # ---------------- UI ----------------
    def _build_ui(self):
        self._window = ui.Window("AgoraSim (B&W)", width=560, height=480)
        self._window.frame.style = VISIBLE_STYLE["Window"]
        with self._window.frame:
            with ui.VStack(spacing=10, height=0):
                ui.Label("AGORASIM", style=VISIBLE_STYLE["Header"])
                self._status_label = ui.Label("Idle", style=VISIBLE_STYLE["StatusBad"])
                self._fps_label = ui.Label("FPS: --", style=VISIBLE_STYLE["Label"])

                with ui.VStack(style=VISIBLE_STYLE["Section"], spacing=6):
                    with ui.HStack(spacing=6):
                        self._run_btn = ui.Button("Start", style=VISIBLE_STYLE["Button"])
                        self._reset_btn = ui.Button("Reset", style=VISIBLE_STYLE["Button"])
                        self._clear_btn = ui.Button("Clear Agents", style=VISIBLE_STYLE["Button"])
                        self._frame_btn = ui.Button("Frame", style=VISIBLE_STYLE["Button"])

                    with ui.HStack(spacing=6):
                        self._spawn_corral_btn = ui.Button("Spawn U-Corral", style=VISIBLE_STYLE["Button"])
                        self._clear_walls_btn = ui.Button("Clear Walls", style=VISIBLE_STYLE["Button"])

                    with ui.HStack(spacing=6):
                        self._spawn_cube_btn = ui.Button("Add Test Cube", style=VISIBLE_STYLE["Button"])
                        self._clear_cubes_btn = ui.Button("Clear Test Cubes", style=VISIBLE_STYLE["Button"])

                    with ui.HStack(spacing=6):
                        self._bake_navmesh_btn = ui.Button("Bake NavMesh", style=VISIBLE_STYLE["Button"])
                        self._clear_navmesh_btn = ui.Button("Clear NavMesh", style=VISIBLE_STYLE["Button"])

                    with ui.HStack(spacing=6):
                        self._spawn_ground_btn = ui.Button("Spawn Ground", style=VISIBLE_STYLE["Button"])
                        self._set_paths_btn = ui.Button("Set Paths (Auto)", style=VISIBLE_STYLE["Button"])
                        self._clear_paths_btn = ui.Button("Clear Paths", style=VISIBLE_STYLE["Button"])

                    with ui.HStack(spacing=4):
                        ui.Label("Agents:", style=VISIBLE_STYLE["Label"])
                        self._agents_int = ui.IntDrag(min=1, max=10000, step=50)
                        self._agents_int.model.set_value(200)

                    with ui.HStack(spacing=4):
                        ui.Label("Speed:", style=VISIBLE_STYLE["Label"])
                        self._speed_slider = ui.FloatSlider(min=0.0, max=20.0)
                        self._speed_slider.model.set_value(6.0)
                        self._speed_display = ui.Label("6.0", style=VISIBLE_STYLE["Label"])

                    with ui.HStack(spacing=4):
                        ui.Label("Target X:", style=VISIBLE_STYLE["Label"])
                        self._target_x = ui.FloatDrag(step=0.5); self._target_x.model.set_value(0.0)
                        ui.Label("Y:", style=VISIBLE_STYLE["Label"])
                        self._target_y = ui.FloatDrag(step=0.5); self._target_y.model.set_value(10.0)

                    with ui.HStack(spacing=4):
                        ui.Label("Arrive Radius:", style=VISIBLE_STYLE["Label"])
                        self._arrive_radius_slider = ui.FloatSlider(min=0.5, max=10.0)
                        self._arrive_radius_slider.model.set_value(3.0)
                        self._arrive_radius_display = ui.Label("3.0", style=VISIBLE_STYLE["Label"])

                self._run_btn.set_clicked_fn(self._toggle_run)
                self._reset_btn.set_clicked_fn(self._reset_clicked)
                self._clear_btn.set_clicked_fn(self._clear_clicked)
                self._frame_btn.set_clicked_fn(self._frame_agents)
                self._spawn_corral_btn.set_clicked_fn(self._spawn_u_corral)
                self._clear_walls_btn.set_clicked_fn(self._clear_walls)
                self._spawn_cube_btn.set_clicked_fn(self._spawn_test_cube)
                self._clear_cubes_btn.set_clicked_fn(self._clear_test_cubes)
                self._spawn_ground_btn.set_clicked_fn(self._spawn_ground_plane)
                self._bake_navmesh_btn.set_clicked_fn(self._bake_navmesh)
                self._clear_navmesh_btn.set_clicked_fn(self._clear_navmesh)
                self._set_paths_btn.set_clicked_fn(self._generate_agent_paths)
                self._clear_paths_btn.set_clicked_fn(self._clear_all_paths)

    # ---------------- Stage / Physics helpers ----------------
    def _ensure_stage(self):
        ctx = omni.usd.get_context()
        stage = ctx.get_stage()
        if not stage:
            ctx.new_stage()
            stage = ctx.get_stage()
        if not stage.GetPrimAtPath("/World"):
            UsdGeom.Xform.Define(stage, "/World")
        return stage

    def _ensure_physics_scene(self):
        """Create a PhysicsScene if missing (required for scene queries & collisions)."""
        stage = self._ensure_stage()
        scene_path = "/World/PhysicsScene"
        if not stage.GetPrimAtPath(scene_path):
            UsdPhysics.Scene.Define(stage, scene_path)
            try:
                scene = UsdPhysics.Scene.Get(stage, scene_path)
                scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
                scene.CreateGravityMagnitudeAttr().Set(981.0)
            except Exception:
                pass

    def _ensure_point_instancer(self, stage, num_agents):
        """
        Ensures:
        - /World/Prototypes/AgentSphere (the single sphere prototype)
        - /World/AgentsPI (UsdGeom.PointInstancer)
        - Sets prototypesRel, positions, orientations, scales, protoIndices
        """
        # 1) Ensure prototype sphere exists
        proto_path = "/World/Prototypes/AgentSphere"
        if not stage.GetPrimAtPath("/World/Prototypes"):
            UsdGeom.Xform.Define(stage, "/World/Prototypes")

        if not stage.GetPrimAtPath(proto_path):
            sphere = UsdGeom.Sphere.Define(stage, proto_path)
            sphere.GetRadiusAttr().Set(0.5)  # default radius
            # Optionally add color
            sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])

        # 2) Ensure PointInstancer exists
        pi_path = "/World/AgentsPI"
        pi_prim = stage.GetPrimAtPath(pi_path)
        if not pi_prim or not pi_prim.IsValid():
            pi_prim = stage.DefinePrim(pi_path, "PointInstancer")

        pi = UsdGeom.PointInstancer(pi_prim)

        # 3) Set prototypes relationship
        pi.CreatePrototypesRel().SetTargets([Sdf.Path(proto_path)])

        # 4) Initialize attributes for num_agents
        # Positions: start at origin (will be updated per-frame)
        positions = [Gf.Vec3f(0.0, 0.0, 0.0)] * num_agents
        pi.GetPositionsAttr().Set(positions)

        # Orientations: identity quaternions
        orientations = [Gf.Quath(1.0, 0.0, 0.0, 0.0)] * num_agents
        pi.GetOrientationsAttr().Set(orientations)

        # Scales: uniform scale of 1
        scales = [Gf.Vec3f(1.0, 1.0, 1.0)] * num_agents
        pi.GetScalesAttr().Set(scales)

        # ProtoIndices: all agents use prototype index 0
        proto_indices = [0] * num_agents
        pi.CreateProtoIndicesAttr().Set(proto_indices)

        # CORRECT WAY: Use PrimvarsAPI to create and set per-instance color
        primvars_api = UsdGeom.PrimvarsAPI(pi.GetPrim())
        color_primvar = primvars_api.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray)
        color_primvar.SetInterpolation(UsdGeom.Tokens.constant) # One color per instance

        # Initialize all to red by default
        colors = [Gf.Vec3f(1.0, 0.0, 0.0)] * num_agents
        color_primvar.Set(colors)

        self._pi = pi
        return pi

    def _frame_agents(self):
        """Frame the camera to show all agents."""
        import omni
        frame_path = "/World/AgentsPI"
        print(f"[AgoraSim] Framing {frame_path}...")

        # 1. Ensure prim exists
        stage = omni.usd.get_context().get_stage()
        if not stage or not stage.GetPrimAtPath(frame_path):
            print(f"[AgoraSim] WARN: Prim {frame_path} not found")
            self._update_status("Agents not found", "error")
            return
        # 2. Select prim via USD selection API (avoid missing commands)
        try:
            sel = omni.usd.get_context().get_selection()
            if hasattr(sel, 'set_selected_prim_paths'):
                sel.set_selected_prim_paths([frame_path], True)
            elif hasattr(sel, 'set_prim_paths'):
                sel.set_prim_paths([frame_path], True)
            print(f"[AgoraSim] Selected {frame_path} via USD selection API")
        except Exception as e:
            print(f"[AgoraSim] WARN: USD selection failed: {e}")

        # 3. Frame using viewport utility
        try:
            from omni.kit.viewport.utility import get_active_viewport
            vp = get_active_viewport()
            if vp and hasattr(vp, 'frame_viewport_selection'):
                vp.frame_viewport_selection()
                print("[AgoraSim] Framed using viewport selection helper")
                self._update_status("Framed agents", "good")
                return
            else:
                print("[AgoraSim] WARN: Active viewport or frame_viewport_selection unavailable")
        except Exception as e:
            print(f"[AgoraSim] WARN: Viewport framing failed: {e}")

        # 4. Last resort: notify user to press F manually
        self._update_status("Select done; press F to frame", "bad")

    def _set_translate_vec3d(self, prim: Usd.Prim, x, y, z=0.0):
        vec = Gf.Vec3d(float(x), float(y), float(z))
        try:
            api = UsdGeom.XformCommonAPI(prim)
            api.SetTranslate(vec)
            return
        except Exception:
            pass
        try:
            attr = prim.GetAttribute("xformOp:translate")
            if attr and attr.IsValid():
                try:
                    attr.Set(vec)
                    return
                except Exception:
                    pass
        except Exception:
            pass
        try:
            xformable = UsdGeom.Xformable(prim)
            existing = prim.GetAttribute("xformOp:translate")
            if not (existing and existing.IsValid()):
                existing = prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3)
            existing.Set(vec)
            return
        except Exception:
            return

    def _apply_collision_api_if_missing(self, prim: Usd.Prim):
        """Idempotently apply CollisionAPI to the given prim (usually a geom prim)."""
        try:
            if not prim or not prim.IsValid():
                return
            if not UsdPhysics.CollisionAPI(prim):
                UsdPhysics.CollisionAPI.Apply(prim)
        except Exception as e:
            print(f"[AgoraSim] WARN: failed to apply CollisionAPI to {prim.GetPath() if prim else prim}: {e}")

    def _get_rigid_body_obstacles(self):
        """
        Find all rigid body cubes in the scene and return their collision bounds.
        Returns list of (min_x, max_x, min_y, max_y) tuples for 2D collision detection.
        """
        obstacles = []

        try:
            stage = omni.usd.get_context().get_stage()
            if not stage:
                return obstacles

            # Traverse all prims in the scene
            for prim in stage.Traverse():
                # Skip agent prims and wall prims
                prim_path = str(prim.GetPath())
                if "/Agents/" in prim_path or "/Walls/" in prim_path:
                    continue

                # Check if this prim has a rigid body API
                if UsdPhysics.RigidBodyAPI(prim):
                    # Look for cube geometry in this prim or its children
                    cube_prim = None

                    # Check if this prim itself is a cube
                    if prim.GetTypeName() == "Cube":
                        cube_prim = prim
                    else:
                        # Check children for cube geometry
                        for child in prim.GetChildren():
                            if child.GetTypeName() == "Cube":
                                cube_prim = child
                                break

                    if cube_prim:
                        # Get the world transform and size of the cube
                        try:
                            # Get cube size (default is 2x2x2, but could be scaled)
                            cube_geom = UsdGeom.Cube(cube_prim)
                            size = cube_geom.GetSizeAttr().Get() or 2.0  # Default cube size
                            half_size = size / 2.0

                            # Get world transform of the cube prim or its parent
                            transform_prim = cube_prim
                            if UsdPhysics.RigidBodyAPI(prim) and prim != cube_prim:
                                transform_prim = prim  # Use rigid body prim for transform

                            # Get world matrix
                            xformable = UsdGeom.Xformable(transform_prim)
                            world_matrix = xformable.ComputeLocalToWorldTransform(0.0)  # Time = 0

                            # Extract translation and scale from matrix
                            translation = world_matrix.ExtractTranslation()
                            scale = Gf.Vec3d(1.0, 1.0, 1.0)

                            # Try to get scale from transform
                            try:
                                transform = world_matrix.RemoveScaleShear()
                                scale_matrix = world_matrix * transform.GetInverse()
                                scale = Gf.Vec3d(
                                    scale_matrix.Transform(Gf.Vec3d(1, 0, 0)).GetLength(),
                                    scale_matrix.Transform(Gf.Vec3d(0, 1, 0)).GetLength(),
                                    scale_matrix.Transform(Gf.Vec3d(0, 0, 1)).GetLength()
                                )
                            except:
                                # Fallback: try to get scale from xform API
                                try:
                                    api = UsdGeom.XformCommonAPI(transform_prim)
                                    scale_vec = api.GetScale()
                                    if scale_vec:
                                        scale = Gf.Vec3d(scale_vec[0], scale_vec[1], scale_vec[2])
                                except:
                                    pass

                            # Calculate actual half-sizes in world space
                            half_x = half_size * scale[0]
                            half_y = half_size * scale[1]

                            # Calculate bounds in 2D (X-Y plane)
                            center_x = translation[0]
                            center_y = translation[1]

                            min_x = center_x - half_x
                            max_x = center_x + half_x
                            min_y = center_y - half_y
                            max_y = center_y + half_y

                            obstacles.append((min_x, max_x, min_y, max_y))

                        except Exception as e:
                            print(f"[AgoraSim] WARN: Failed to get bounds for cube {prim_path}: {e}")

        except Exception as e:
            print(f"[AgoraSim] WARN: Error finding rigid body obstacles: {e}")

        return obstacles

    def _spawn_ground_plane(self):
        """Create a large horizontal ground plane for NavMesh baking."""
        stage = self._ensure_stage()
        ground_path = "/World/Ground"

        # Remove old ground if exists
        if stage.GetPrimAtPath(ground_path):
            stage.RemovePrim(ground_path)

        # Create a large quad mesh - horizontal plane BELOW agents
        # Agents are at Z=0.5, so ground at Z=0 is beneath them
        mesh = UsdGeom.Mesh.Define(stage, ground_path)

        # 100x100 unit plane (covers -50 to +50 in X and Y)
        # Z=0 makes it horizontal and below the agents (at Z=0.5)
        mesh.CreatePointsAttr([
            Gf.Vec3f(-50, -50, 0),  # Bottom-left corner
            Gf.Vec3f(50, -50, 0),   # Bottom-right corner
            Gf.Vec3f(50, 50, 0),    # Top-right corner
            Gf.Vec3f(-50, 50, 0)    # Top-left corner
        ])

        # Two triangles forming a horizontal quad
        # Triangle 1: vertices 0,1,2 | Triangle 2: vertices 0,2,3
        mesh.CreateFaceVertexCountsAttr([3, 3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])

        # Normals pointing UP (positive Z direction) - this makes it a floor, not a wall
        mesh.CreateNormalsAttr([
            Gf.Vec3f(0, 0, 1),  # Point 0 normal: UP
            Gf.Vec3f(0, 0, 1),  # Point 1 normal: UP
            Gf.Vec3f(0, 0, 1),  # Point 2 normal: UP
            Gf.Vec3f(0, 0, 1)   # Point 3 normal: UP
        ])

        # Light gray color for visibility
        mesh.CreateDisplayColorAttr([Gf.Vec3f(0.5, 0.5, 0.5)])

        print(f"[AgoraSim] Created HORIZONTAL ground plane at {ground_path}")
        print(f"[AgoraSim]   Dimensions: 100x100 units (X: -50 to +50, Y: -50 to +50, Z: 0)")
        print(f"[AgoraSim]   Agents at Z=0.5 are ABOVE the ground at Z=0")
        self._update_status("Ground plane spawned", "good")

    def _spawn_u_corral(self):
        """
        Spawns a simple U-shaped set of walls with static colliders at /World/Walls:
          - Bottom: width 40, depth 2
          - Left & Right: height 20, depth 2
        Agents should be stopped by raycast gating when moving into these walls.
        """
        stage = self._ensure_stage()
        walls_root = "/World/Walls"
        if not stage.GetPrimAtPath(walls_root):
            UsdGeom.Xform.Define(stage, walls_root)

        def _mk_wall(name, center_xyz, scale_xyz):
            wall_xf = UsdGeom.Xform.Define(stage, f"{walls_root}/{name}")
            api = UsdGeom.XformCommonAPI(wall_xf)
            api.SetTranslate(Gf.Vec3d(*center_xyz))
            api.SetScale(Gf.Vec3f(*scale_xyz))

            cube = UsdGeom.Cube.Define(stage, f"{walls_root}/{name}/Cube")
            cube.GetDisplayColorAttr().Set([Gf.Vec3f(0.25, 0.25, 0.25)])
            # Apply collision to the Cube geom prim
            self._apply_collision_api_if_missing(cube.GetPrim())
            return wall_xf

        # Dimensions are applied via parent Xform scale (Cube default size is 2x2x2 in USD)
        # so scale roughly half of the desired absolute size (since cube size=2).
        # Choose Z=1.0 so top is at z=1.0 (agent z=0.5).
        _mk_wall("Bottom", center_xyz=(0.0, -10.0, 1.0), scale_xyz=(20.0, 1.0, 1.0))  # width ~40, thickness ~2
        _mk_wall("Left",   center_xyz=(-20.0, 0.0, 1.0), scale_xyz=(1.0, 10.0, 1.0))  # height ~20
        _mk_wall("Right",  center_xyz=(20.0,  0.0, 1.0), scale_xyz=(1.0, 10.0, 1.0))  # height ~20

        print("[AgoraSim] Spawned U-Corral at /World/Walls")
        self._update_status("Corral spawned", "good")

    def _clear_walls(self):
        stage = self._ensure_stage()
        walls_root = "/World/Walls"
        if stage.GetPrimAtPath(walls_root):
            stage.RemovePrim(walls_root)
            print("[AgoraSim] Cleared walls")
        self._update_status("Walls cleared", "bad")

    def _spawn_test_cube(self):
        """Spawn a rigid body cube obstacle for testing collision detection."""
        stage = self._ensure_stage()

        # Create unique name for the cube
        cube_index = 0
        while stage.GetPrimAtPath(f"/World/TestCube_{cube_index}"):
            cube_index += 1

        cube_path = f"/World/TestCube_{cube_index}"

        # Create xform prim for the rigid body
        cube_xform = UsdGeom.Xform.Define(stage, cube_path)

        # Position randomly in the scene (avoid agent spawn area)
        import random
        x = random.uniform(-15, 15)
        y = random.uniform(-5, 15)
        z = 1.0  # Above ground

        # Set position
        api = UsdGeom.XformCommonAPI(cube_xform)
        api.SetTranslate(Gf.Vec3d(x, y, z))
        api.SetScale(Gf.Vec3f(2.0, 2.0, 2.0))  # Make it 4x4x4 units

        # Create cube geometry
        cube_geom = UsdGeom.Cube.Define(stage, f"{cube_path}/CubeGeom")
        cube_geom.GetDisplayColorAttr().Set([Gf.Vec3f(0.8, 0.3, 0.3)])  # Red color

        # Apply physics APIs
        try:
            # Apply RigidBodyAPI to the xform prim
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(cube_xform.GetPrim())
            rigid_body_api.CreateRigidBodyEnabledAttr(True)

            # Apply CollisionAPI to the geometry prim
            collision_api = UsdPhysics.CollisionAPI.Apply(cube_geom.GetPrim())

            # Make it static (kinematic) so it doesn't fall due to gravity
            mass_api = UsdPhysics.MassAPI.Apply(cube_xform.GetPrim())
            mass_api.CreateMassAttr(0.0)  # Zero mass = kinematic/static

            print(f"[AgoraSim] Spawned test cube at ({x:.1f}, {y:.1f}, {z:.1f})")
            self._update_status(f"Test cube spawned at ({x:.1f}, {y:.1f})", "good")

        except Exception as e:
            print(f"[AgoraSim] ERROR: Failed to create rigid body cube: {e}")
            self._update_status("Failed to create test cube", "error")

    def _clear_test_cubes(self):
        """Remove all test cubes from the scene."""
        stage = self._ensure_stage()
        removed_count = 0

        try:
            # Find and remove all TestCube_* prims
            for prim in list(stage.Traverse()):
                prim_path = str(prim.GetPath())
                if "/TestCube_" in prim_path and prim.GetPath().GetParentPath() == "/World":
                    stage.RemovePrim(prim.GetPath())
                    removed_count += 1

            if removed_count > 0:
                print(f"[AgoraSim] Cleared {removed_count} test cubes")
                self._update_status(f"Cleared {removed_count} test cubes", "bad")
            else:
                print("[AgoraSim] No test cubes to clear")
                self._update_status("No test cubes found", "bad")

        except Exception as e:
            print(f"[AgoraSim] ERROR: Failed to clear test cubes: {e}")
            self._update_status("Failed to clear test cubes", "error")

    # ---------------- Path Following Methods ----------------
    def _generate_agent_paths(self):
        """
        Generate paths for all agents using NavMesh.
        If a NavMesh path is not found, it creates a direct path to the goal as a fallback.
        """
        if self._num_agents == 0:
            print("[AgoraSim] No agents to set paths for.")
            self._update_status("No agents", "error")
            return

        if not _HAS_ISAAC_NAV or self._navmesh_handle is None:
            print("[AgoraSim] WARN: NavMesh not available or not baked. Paths will be direct.")
            # Do not simply return; proceed with direct path fallback.

        print(f"[AgoraSim] Generating paths for {self._num_agents} agents...")

        # Get current agent positions
        if not _HAS_WARP or self._positions_wp is None:
            print("[AgoraSim] ERROR: No agent positions available to generate paths from.")
            self._update_status("Agent data missing", "error")
            return

        current_positions = self._positions_wp.numpy()

        import random
        self._agent_paths = []
        all_waypoints = []
        waypoint_indices = []
        current_waypoint_offset = 0

        paths_found = 0
        paths_failed = 0
        agent_colors = []

        for i in range(self._num_agents):
            start_pos = (current_positions[i][0], current_positions[i][1])
            goal_x = random.uniform(-20, 20)
            goal_y = random.uniform(-10, 20)
            goal_pos = (goal_x, goal_y)

            # Always attempt to find a path on the NavMesh
            path = self._find_path_on_navmesh(start_pos, goal_pos)

            if path and len(path) > 1:
                # NavMesh path found. Use it, skipping the first waypoint (current position).
                path_waypoints = path[1:]
                self._agent_paths.append(path_waypoints)
                all_waypoints.extend(path_waypoints)
                waypoint_indices.append(current_waypoint_offset)
                current_waypoint_offset += len(path_waypoints)
                paths_found += 1
                agent_colors.append(Gf.Vec3f(1.0, 0.0, 0.0)) # Red for success
            else:
                # Fallback: No path found. Agent will be stationary.
                self._agent_paths.append([])
                waypoint_indices.append(-1) # -1 indicates no path
                paths_failed += 1
                agent_colors.append(Gf.Vec3f(1.0, 1.0, 0.0)) # Yellow for failure

        # Convert to Warp arrays
        if _HAS_WARP:
            dev = wp.get_preferred_device()

            if len(all_waypoints) > 0:
                waypoints_flat = np.array(all_waypoints, dtype=np.float32)
                # Construct as 1D array of vec2 (each row -> wp.vec2)
                self._waypoints_wp = wp.array(waypoints_flat, dtype=wp.vec2, device=dev)
            else:
                # Zero waypoints: create proper empty 1D vec2 buffer
                self._waypoints_wp = wp.empty(shape=(0,), dtype=wp.vec2, device=dev)

            waypoint_indices_np = np.array(waypoint_indices, dtype=np.int32)
            self._agent_waypoint_idx = wp.array(waypoint_indices_np, dtype=wp.int32, device=dev)

            # Update agent colors based on path status
            if self._pi:
                primvars_api = UsdGeom.PrimvarsAPI(self._pi.GetPrim())
                color_primvar = primvars_api.GetPrimvar("displayColor")
                if color_primvar:
                    color_primvar.Set(agent_colors)

            print(f"[AgoraSim] Path generation complete. NavMesh paths: {paths_found}, No paths: {paths_failed}")
            self._update_status(f"Paths set ({paths_found} NavMesh)", "good")
        else:
            print("[AgoraSim] Failed to create any paths.")
            self._update_status("Path generation failed", "error")

    def _clear_all_paths(self):
        """Clear all agent paths."""
        self._agent_paths = []

        if _HAS_WARP and self._agent_waypoint_idx is not None:
            # Set all waypoint indices to -1 (no path)
            dev = wp.get_preferred_device()
            no_paths = np.full(self._num_agents, -1, dtype=np.int32)
            self._agent_waypoint_idx = wp.array(no_paths, dtype=wp.int32, device=dev)

        print("[AgoraSim] Cleared all agent paths")
        self._update_status("Paths cleared", "bad")

    def _update_agent_waypoints(self, positions_np):
        """
        Check if agents have reached their current waypoint and advance to next.
        Returns updated waypoint indices.
        """
        if not _HAS_WARP or self._agent_waypoint_idx is None or self._waypoints_wp is None:
            return

        # Get current state from GPU
        waypoint_indices = self._agent_waypoint_idx.numpy()
        all_waypoints = self._waypoints_wp.numpy()

        agents_reached_goal = 0
        waypoints_advanced = 0

        for i in range(self._num_agents):
            current_wp_idx = waypoint_indices[i]

            # Skip if agent has no path
            if current_wp_idx < 0 or current_wp_idx >= len(all_waypoints):
                continue

            # Get agent position and current waypoint
            agent_pos = positions_np[i]
            waypoint = all_waypoints[current_wp_idx]

            # Check distance to waypoint
            dx = waypoint[0] - agent_pos[0]
            dy = waypoint[1] - agent_pos[1]
            dist = np.sqrt(dx*dx + dy*dy)

            # If close enough, advance to next waypoint
            if dist < self._waypoint_reach_distance:
                agent_path = self._agent_paths[i]

                # Find current waypoint position in this agent's path
                path_start_idx = 0
                for prev_agent in range(i):
                    path_start_idx += len(self._agent_paths[prev_agent])

                current_waypoint_in_path = current_wp_idx - path_start_idx

                # Check if there's a next waypoint
                if current_waypoint_in_path + 1 < len(agent_path):
                    # Advance to next waypoint
                    waypoint_indices[i] = current_wp_idx + 1
                    waypoints_advanced += 1
                else:
                    # Reached end of path - stop (set to -1)
                    waypoint_indices[i] = -1
                    agents_reached_goal += 1

        # Update GPU array
        if waypoints_advanced > 0 or agents_reached_goal > 0:
            dev = wp.get_preferred_device()
            self._agent_waypoint_idx = wp.array(waypoint_indices, dtype=wp.int32, device=dev)

            if agents_reached_goal > 0:
                print(f"[AgoraSim] {agents_reached_goal} agents reached their goal")

    # ---------------- NavMesh Methods ----------------
    def _init_navmesh_config(self):
        """Initialize Isaac Sim navigation parameters."""
        self._navmesh_config = {
            # Voxelization
            'cell_size': 0.15,          # XY voxel size (m)
            'cell_height': 0.1,         # Z voxel size (m)

            # Agent properties (match your sim)
            'agent_radius': 0.6,        # Match agent collision radius
            'agent_height': 1.8,        # Typical human height
            'agent_max_climb': 0.4,     # Max step height (m)
            'agent_max_slope': 45.0,    # Max walkable slope (degrees)

            # Region partitioning
            'tile_size': 48,            # Cells per tile

            # Polygonization
            'region_min_size': 8,       # Min region size (cells²)
            'region_merge_size': 20,    # Merge threshold
            'edge_max_len': 12.0,       # Max edge length (cells)
            'edge_max_error': 1.3,      # Edge simplification tolerance
        }

    def _collect_walkable_geometry(self):
        """Extract floor/ramp geometry from USD stage for navmesh baking."""
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return []

        walkable_meshes = []

        # Traverse stage looking for walkable geometry
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())

            # Include floors, exclude walls/agents/prototypes
            if any(x in prim_path for x in ["/Agents", "/Walls", "/Prototypes", "/TestCube"]):
                continue

            # Look for geometry marked as walkable
            if prim.GetTypeName() in ["Mesh", "Plane", "Cube"]:
                # Check if marked as walkable (custom attribute or naming convention)
                prim_name = prim_path.lower()
                if any(keyword in prim_name for keyword in ["floor", "ground", "walkable", "navmesh"]):
                    mesh_data = self._extract_mesh_data(prim)
                    if mesh_data:
                        walkable_meshes.append(mesh_data)
                        print(f"[AgoraSim] Found walkable geometry: {prim_path}")

        return walkable_meshes

    def _extract_mesh_data(self, prim):
        """Extract vertices and triangles from USD geom prim."""
        try:
            prim_type = prim.GetTypeName()

            if prim_type == "Mesh":
                mesh = UsdGeom.Mesh(prim)
                points = mesh.GetPointsAttr().Get()
                indices = mesh.GetFaceVertexIndicesAttr().Get()
            elif prim_type == "Plane":
                # Generate plane mesh data
                # Planes in USD are typically 2x2 units centered at origin
                points = [
                    Gf.Vec3f(-1, -1, 0),
                    Gf.Vec3f(1, -1, 0),
                    Gf.Vec3f(1, 1, 0),
                    Gf.Vec3f(-1, 1, 0)
                ]
                indices = [0, 1, 2, 0, 2, 3]  # Two triangles
            elif prim_type == "Cube":
                # Use top face of cube for walkable surface
                points = [
                    Gf.Vec3f(-1, -1, 1),
                    Gf.Vec3f(1, -1, 1),
                    Gf.Vec3f(1, 1, 1),
                    Gf.Vec3f(-1, 1, 1)
                ]
                indices = [0, 1, 2, 0, 2, 3]
            else:
                return None

            if not points or not indices:
                return None

            # Transform to world space
            xformable = UsdGeom.Xformable(prim)
            world_matrix = xformable.ComputeLocalToWorldTransform(0.0)

            world_points = [world_matrix.Transform(Gf.Vec3d(p)) for p in points]

            return {
                'vertices': world_points,
                'indices': list(indices),
                'prim_path': str(prim.GetPath())
            }
        except Exception as e:
            print(f"[AgoraSim] Failed to extract mesh from {prim.GetPath()}: {e}")
            return None

    def _bake_navmesh(self):
        """Bake navigation mesh from scene geometry using Isaac Sim."""
        if not _HAS_ISAAC_NAV:
            self._update_status("Isaac Nav not available", "error")
            print("[AgoraSim] ERROR: Isaac Sim navigation not available")
            return

        print("[AgoraSim] Baking NavMesh...")
        self._update_status("Baking NavMesh...", "good")

        try:
            # 1. Collect walkable geometry
            walkable_meshes = self._collect_walkable_geometry()
            if not walkable_meshes:
                print("[AgoraSim] WARN: No walkable geometry found. Add a prim with 'floor' or 'ground' in its name.")
                self._update_status("No walkable geometry found", "error")
                return

            print(f"[AgoraSim] Found {len(walkable_meshes)} walkable meshes")

            # 2. Get or create navigation volume
            stage = self._ensure_stage()
            nav_volume_path = "/World/NavMeshVolume"

            # Remove old volume if exists
            if stage.GetPrimAtPath(nav_volume_path):
                stage.RemovePrim(nav_volume_path)

            # 3. Create NavMesh volume using Isaac API
            # Note: Isaac Sim's API may vary - this is the conceptual approach
            try:
                # Create navigation volume prim
                nav_volume_prim = stage.DefinePrim(nav_volume_path, "Volume")

                # Set up navigation parameters
                # Isaac Sim uses USD schema attributes for navigation config
                try:
                    nav_volume_prim.CreateAttribute("navmesh:cellSize", Sdf.ValueTypeNames.Float).Set(self._navmesh_config['cell_size'])
                    nav_volume_prim.CreateAttribute("navmesh:cellHeight", Sdf.ValueTypeNames.Float).Set(self._navmesh_config['cell_height'])
                    nav_volume_prim.CreateAttribute("navmesh:agentRadius", Sdf.ValueTypeNames.Float).Set(self._navmesh_config['agent_radius'])
                    nav_volume_prim.CreateAttribute("navmesh:agentHeight", Sdf.ValueTypeNames.Float).Set(self._navmesh_config['agent_height'])
                    nav_volume_prim.CreateAttribute("navmesh:agentMaxClimb", Sdf.ValueTypeNames.Float).Set(self._navmesh_config['agent_max_climb'])
                    nav_volume_prim.CreateAttribute("navmesh:agentMaxSlope", Sdf.ValueTypeNames.Float).Set(self._navmesh_config['agent_max_slope'])
                except Exception as attr_err:
                    print(f"[AgoraSim] WARN: Failed setting navmesh config attributes: {attr_err}")

                # 4. Actually bake the NavMesh using nav_core API
                print("[AgoraSim] NavMesh volume created at", nav_volume_path)

                # Prepare mesh data for nav_core
                try:
                    # Collect all vertices and indices from walkable meshes
                    all_vertices = []
                    all_indices = []
                    vertex_offset = 0

                    for mesh in walkable_meshes:
                        vertices = mesh['vertices']
                        indices = mesh['indices']

                        # Add vertices
                        for v in vertices:
                            all_vertices.extend([v[0], v[1], v[2]])

                        # Add indices with offset
                        for idx in indices:
                            all_indices.append(idx + vertex_offset)

                        vertex_offset += len(vertices)

                    print(f"[AgoraSim] Prepared {len(all_vertices)//3} vertices, {len(all_indices)//3} triangles")

                    # Actually create and bake the navmesh using Isaac Sim navigation interface
                    try:
                        # Clear old navmesh handle if exists
                        if self._navmesh_handle is not None:
                            try:
                                if hasattr(self._navmesh_handle, 'destroy'):
                                    self._navmesh_handle.destroy()
                                elif hasattr(self._navmesh_handle, 'release'):
                                    self._navmesh_handle.release()
                            except Exception:
                                pass
                            self._navmesh_handle = None

                        # Get the navigation interface (INavigation)
                        nav_interface = nav_core.acquire_interface()
                        print(f"[AgoraSim] Acquired navigation interface: {nav_interface}")

                        if nav_interface is None:
                            print("[AgoraSim] ERROR: Failed to acquire navigation interface")
                            self._navmesh_handle = None
                            self._update_status("NavMesh interface failed", "error")
                        else:
                            # The NavMesh volume prim path we created
                            volume_path = nav_volume_path

                            # Use the interface to bake the navmesh
                            # Isaac Sim's nav interface typically works with the USD prim we created
                            print(f"[AgoraSim] Baking navmesh for volume: {volume_path}")

                            try:
                                # Isaac Sim uses start_navmesh_baking_and_wait() to bake synchronously
                                if hasattr(nav_interface, 'start_navmesh_baking_and_wait'):
                                    print(f"[AgoraSim] Starting synchronous navmesh baking for {volume_path}...")
                                    nav_interface.start_navmesh_baking_and_wait(volume_path)
                                    print("[AgoraSim] NavMesh baking completed")
                                elif hasattr(nav_interface, 'start_navmesh_baking'):
                                    print(f"[AgoraSim] Starting async navmesh baking for {volume_path}...")
                                    nav_interface.start_navmesh_baking(volume_path)
                                    print("[AgoraSim] NavMesh baking started (async)")
                                else:
                                    print("[AgoraSim] ERROR: No baking method found on navigation interface")
                                    print("[AgoraSim] Available methods:", [m for m in dir(nav_interface) if not m.startswith('_')])
                                    self._navmesh_handle = None
                                    self._update_status("No baking method found", "error")
                                    return

                                # Get the actual navmesh handle from the interface
                                if hasattr(nav_interface, 'get_navmesh'):
                                    navmesh_handle = nav_interface.get_navmesh(volume_path)
                                    if navmesh_handle:
                                        self._navmesh_handle = navmesh_handle
                                        print(f"[AgoraSim] NavMesh handle acquired: {navmesh_handle}")
                                    else:
                                        print("[AgoraSim] WARN: get_navmesh returned None")
                                        self._navmesh_handle = volume_path  # Fallback to volume path
                                else:
                                    # Fallback: store volume path and query via interface
                                    self._navmesh_handle = volume_path
                                    print("[AgoraSim] No get_navmesh method - using volume path as handle")

                                self._update_status(f"NavMesh baked ({len(walkable_meshes)} meshes)", "good")
                                print(f"[AgoraSim] NavMesh ready at: {volume_path}")
                                print("[AgoraSim] TIP: Enable Viewport → Visibility → Show by Type → Navmesh to visualize")

                            except Exception as bake_err:
                                print(f"[AgoraSim] ERROR during navmesh baking: {bake_err}")
                                print("[AgoraSim] DEBUG: nav_interface type:", type(nav_interface))
                                print("[AgoraSim] DEBUG: nav_interface methods:", dir(nav_interface))
                                self._navmesh_handle = None
                                self._update_status("NavMesh bake failed", "error")

                    except Exception as nav_err:
                        print(f"[AgoraSim] ERROR acquiring navigation interface: {nav_err}")
                        import traceback
                        traceback.print_exc()
                        self._navmesh_handle = None
                        self._update_status("NavMesh interface error", "error")

                except Exception as prep_err:
                    print(f"[AgoraSim] Error preparing mesh data: {prep_err}")
                    self._update_status("Mesh data preparation failed", "error")

            except Exception as e:
                print(f"[AgoraSim] ERROR creating NavMesh volume: {e}")
                self._update_status("NavMesh volume creation failed", "error")

        except Exception as e:
            print(f"[AgoraSim] ERROR during NavMesh baking: {e}")
            self._update_status("NavMesh bake error", "error")

    def _clear_navmesh(self):
        """Remove the navigation mesh from the scene."""
        # Release navmesh handle first
        if self._navmesh_handle is not None:
            try:
                if hasattr(self._navmesh_handle, 'destroy'):
                    self._navmesh_handle.destroy()
                elif hasattr(self._navmesh_handle, 'release'):
                    self._navmesh_handle.release()
                print("[AgoraSim] Released NavMesh handle")
            except Exception as e:
                print(f"[AgoraSim] WARN: Failed to release navmesh handle: {e}")
            self._navmesh_handle = None

        # Remove USD prim
        stage = self._ensure_stage()
        nav_volume_path = "/World/NavMeshVolume"

        if stage.GetPrimAtPath(nav_volume_path):
            stage.RemovePrim(nav_volume_path)
            print("[AgoraSim] Cleared NavMesh volume")
            self._update_status("NavMesh cleared", "bad")
        else:
            print("[AgoraSim] No NavMesh to clear")
            self._update_status("No NavMesh found", "bad")

        self._navmesh_volume = None

    def _find_path_on_navmesh(self, start_pos, end_pos):
        """Find path using Isaac navigation interface."""
        if not _HAS_ISAAC_NAV or self._navmesh_handle is None:
            return None

        try:
            # Get the navigation interface
            nav_interface = nav_core.acquire_interface()
            if nav_interface is None:
                return None

            # Convert 2D positions to 3D (NavMesh uses 3D)
            start_3d = [float(start_pos[0]), float(start_pos[1]), 0.5]
            goal_3d = [float(end_pos[0]), float(end_pos[1]), 0.5]

            # Try to find path using the navmesh handle
            path_result = None

            # If handle is a navmesh object, it might have find_path
            if hasattr(self._navmesh_handle, 'find_path'):
                path_result = self._navmesh_handle.find_path(start_3d, goal_3d)
            # Otherwise try the interface with the handle (could be volume path or navmesh object)
            elif hasattr(nav_interface, 'find_path'):
                path_result = nav_interface.find_path(self._navmesh_handle, start_3d, goal_3d)
            elif hasattr(nav_interface, 'query_path'):
                path_result = nav_interface.query_path(self._navmesh_handle, start_3d, goal_3d)
            elif hasattr(nav_interface, 'get_path'):
                path_result = nav_interface.get_path(self._navmesh_handle, start_3d, goal_3d)
            else:
                # Debug: show what methods are available (only once)
                if not hasattr(self, '_path_debug_shown'):
                    print("[AgoraSim] DEBUG: Looking for path query methods...")
                    print("[AgoraSim] DEBUG: nav_interface type:", type(nav_interface))
                    path_methods = [m for m in dir(nav_interface) if 'path' in m.lower() and not m.startswith('_')]
                    print("[AgoraSim] DEBUG: Available path-related methods:", path_methods)
                    print("[AgoraSim] DEBUG: navmesh_handle type:", type(self._navmesh_handle))
                    if hasattr(self._navmesh_handle, '__dict__'):
                        handle_methods = [m for m in dir(self._navmesh_handle) if 'path' in m.lower() and not m.startswith('_')]
                        print("[AgoraSim] DEBUG: Handle path methods:", handle_methods)
                    self._path_debug_shown = True
                return None

            if path_result and len(path_result) > 0:
                # Convert 3D path to 2D waypoints (X, Y only)
                waypoints_2d = [(wp[0], wp[1]) for wp in path_result]
                return waypoints_2d
            else:
                # No path found
                return None

        except Exception as e:
            if not hasattr(self, '_path_error_shown'):
                print(f"[AgoraSim] ERROR finding path: {e}")
                import traceback
                traceback.print_exc()
                self._path_error_shown = True
            return None

    def _raycast_navmesh(self, start_pos, end_pos):
        """Check if straight line path is valid on navmesh."""
        if not _HAS_ISAAC_NAV or not self._navmesh_volume:
            return False

        try:
            # Placeholder for Isaac Sim raycast API
            # hit = nav_core.raycast(start_pos, end_pos)
            # return hit is None
            pass
        except Exception as e:
            print(f"[AgoraSim] ERROR raycasting navmesh: {e}")
            return False

    def _update_status(self, text, mode="neutral"):
        style = {
            "good": VISIBLE_STYLE["StatusGood"],
            "bad": VISIBLE_STYLE["StatusBad"],
            "error": VISIBLE_STYLE["StatusError"],
        }.get(mode, VISIBLE_STYLE["StatusBad"])
        self._status_label.text = text
        self._status_label.style = style

    # ---------------- UI callbacks ----------------
    def _toggle_run(self):
        if not _HAS_WARP:
            self._update_status("Warp not available", "error")
            return
        self._is_running = not self._is_running
        self._run_btn.text = "Stop" if self._is_running else "Start"
        self._update_status("Running" if self._is_running else "Stopped",
                            "good" if self._is_running else "bad")
        if self._is_running and self._num_agents == 0:
            self._reset_clicked()

    def _reset_clicked(self):
        self._is_running = False
        self._run_btn.text = "Start"
        n = int(self._agents_int.model.as_int)
        self._reset_simulation(n)

    def _clear_clicked(self):
        stage = self._ensure_stage()
        agents_root_path = "/World/Agents"
        if stage.GetPrimAtPath(agents_root_path):
            stage.RemovePrim(agents_root_path)
        # Also remove PointInstancer if it exists
        pi_path = "/World/AgentsPI"
        if stage.GetPrimAtPath(pi_path):
            stage.RemovePrim(pi_path)
        self._agent_prims = []
        self._positions_wp = None
        self._num_agents = 0
        self._pi = None
        self._update_status("Cleared agents", "bad")

    def _update_point_instancer(self, pos_np):
        """Update PointInstancer positions from numpy array of 2D positions."""
        if not self._pi:
            return

        agent_z = 0.5
        # Convert numpy.float32 to Python float for USD compatibility
        positions_3d = [Gf.Vec3f(float(pos_np[i][0]), float(pos_np[i][1]), agent_z) for i in range(len(pos_np))]
        self._pi.GetPositionsAttr().Set(positions_3d)

    # ---------------- Simulation setup ----------------
    def _reset_simulation(self, num_agents=200):
        stage = self._ensure_stage()

        # Remove old PointInstancer if it exists
        pi_path = "/World/AgentsPI"
        if stage.GetPrimAtPath(pi_path):
            stage.RemovePrim(pi_path)

        # Remove old individual agents if they exist
        agents_root = "/World/Agents"
        if stage.GetPrimAtPath(agents_root):
            stage.RemovePrim(agents_root)

        self._agent_prims.clear()
        self._num_agents = num_agents

        # Generate random starting positions (N,2)
        pos_np = np.zeros((num_agents, 2), dtype=np.float32)
        for i in range(num_agents):
            start = (np.random.rand(2).astype(np.float32) * 50 - 25)
            pos_np[i] = start

        try:
            sample = pos_np[: min(5, num_agents)].tolist()
            print(f"[AgoraSim] Created {num_agents} agents; sample positions: {sample}")
        except Exception:
            pass

        # Initialize Warp buffer
        if _HAS_WARP:
            dev = wp.get_preferred_device()
            self._positions_wp = wp.array(pos_np, dtype=wp.vec2, device=dev)

        # Create PointInstancer
        self._ensure_point_instancer(stage, num_agents)

        # Sync initial positions to PointInstancer
        self._update_point_instancer(pos_np)

        self._update_status(f"Ready • {num_agents} agents (PointInstancer)", "good")

    # ---------------- Timing helper ----------------
    def _safe_get_dt(self, event):
        try:
            if event is not None:
                if hasattr(event, 'dt'): return float(event.dt)
                if hasattr(event, 'deltaTime'): return float(event.deltaTime)
                if hasattr(event, 'delta_in_seconds'): return float(event.delta_in_seconds)
                if hasattr(event, 'delta_seconds'): return float(event.delta_seconds)
                if hasattr(event, 'get_dt'):
                    try: return float(event.get_dt())
                    except Exception: pass
        except Exception:
            pass
        try:
            app = kit_app.get_app()
            if hasattr(app, 'get_time'):
                try: return float(app.get_time().get_dt())
                except Exception: pass
            if hasattr(app, 'get_dt'):
                try: return float(app.get_dt())
                except Exception: pass
        except Exception:
            pass
        return 1.0 / 60.0

    # ---------------- Update loop ----------------
    def _on_update(self, e):
        if not (self._is_running and _HAS_WARP and self._positions_wp is not None):
            return

        self._update_call_count = getattr(self, "_update_call_count", 0) + 1

        # Calculate FPS
        dt = self._safe_get_dt(e)
        if dt > 0:
            fps = 1.0 / dt
            self._fps_history.append(fps)
            if len(self._fps_history) > self._fps_window_size:
                self._fps_history.pop(0)
            avg_fps = sum(self._fps_history) / len(self._fps_history)

            # Update FPS display every 10 frames
            if self._update_call_count % 10 == 0:
                self._fps_label.text = f"FPS: {avg_fps:.1f} (200 agents)"

        # Debug: print every 60 frames to confirm update loop is running
        if self._update_call_count % 60 == 0:
            print(f"[AgoraSim] Update loop running, call count: {self._update_call_count}")

        # 1) Sim params
        try:
            dt_sim = self._safe_get_dt(e)
            if dt_sim is None or dt_sim <= 0 or dt_sim > 0.1:
                dt_sim = 1.0 / 60.0

            self._speed = float(self._speed_slider.model.as_float)
            if getattr(self, "_speed_display", None):
                self._speed_display.text = f"{self._speed:.1f}"

            self._arrive_radius = float(self._arrive_radius_slider.model.as_float)
            if getattr(self, "_arrive_radius_display", None):
                self._arrive_radius_display.text = f"{self._arrive_radius:.1f}"

            current_positions = self._positions_wp.numpy()
            query_interface = get_physx_scene_query_interface()
            if query_interface is None:
                print("[AgoraSim] WARN: PhysX scene query interface not available; skipping collision checks.")

            # Collision system configuration
            # Disable raycast collision (too many false positives from ground/invisible geometry)
            # Use simple agent-to-agent collision instead
            use_collision = False  # Set to True to enable PhysX raycast collision
            agent_radius = 0.6    # Agent collision radius (slightly larger than visual sphere)
        except Exception as exc:
            print(f"[AgoraSim] ERROR: Failed to get UI/sim parameters: {exc}")
            self._update_status("Sim error (params)", "error")
            self._is_running = False
            return

        # 2) Move proposal (GPU) - now using path following
        try:
            # First, update which waypoint each agent is targeting
            self._update_agent_waypoints(current_positions)

            # Check if we have valid path data
            if (self._agent_waypoint_idx is None or
                self._waypoints_wp is None or
                getattr(self._waypoints_wp, 'shape', None) in (None,) or
                (hasattr(self._waypoints_wp, 'shape') and self._waypoints_wp.shape[0] == 0)):
                # No usable paths (all agents idle) - keep positions
                proposed_positions = np.copy(current_positions)
            else:
                # Use path-following kernel
                proposed_positions_wp = wp.empty_like(self._positions_wp)
                wp.copy(proposed_positions_wp, self._positions_wp)
                wp.launch(
                    kernel=self._move_agents,
                    dim=self._num_agents,
                    inputs=[
                        proposed_positions_wp,
                        self._waypoints_wp,
                        self._agent_waypoint_idx,
                        self._speed,
                        dt_sim,
                        self._arrive_radius
                    ],
                    device=self._positions_wp.device
                )
                wp.synchronize()
                proposed_positions = proposed_positions_wp.numpy()
        except Exception as exc:
            print(f"[AgoraSim] ERROR: Warp kernel launch failed: {exc}")
            self._update_status("Sim error (kernel)", "error")
            self._is_running = False
            self._run_btn.text = "Start"
            return

        # 3) Raycast gating (or just use proposed positions if no collision system)
        try:
            final_positions = np.copy(current_positions)
            agent_z = 0.5

            if query_interface is not None and use_collision:
                # Simple collision: if raycast hits anything, don't move
                moves_allowed = 0
                moves_blocked = 0

                for i in range(self._num_agents):
                    current_pos = current_positions[i]
                    proposed_pos = proposed_positions[i]

                    # Calculate movement
                    dx = proposed_pos[0] - current_pos[0]
                    dy = proposed_pos[1] - current_pos[1]
                    move_distance = np.sqrt(dx*dx + dy*dy)

                    if move_distance > 0.001:  # Only check if moving
                        # Raycast from current to proposed position at agent height
                        # Use higher Z to avoid ground plane collisions
                        raycast_height = agent_z + 0.1  # Slightly above agent center
                        origin = carb.Float3(current_pos[0], current_pos[1], raycast_height)
                        direction = carb.Float3(dx/move_distance, dy/move_distance, 0.0)

                        # Raycast the movement distance (horizontal only)
                        hit = query_interface.raycast_closest(origin, direction, move_distance, bothSides=False)

                        if hit is None:
                            # No collision - allow movement
                            final_positions[i] = proposed_pos
                            moves_allowed += 1
                        else:
                            # Hit something - don't move
                            final_positions[i] = current_pos
                            moves_blocked += 1
                    else:
                        # Not moving much, allow it
                        final_positions[i] = proposed_pos
                        moves_allowed += 1

                # Debug every 5 seconds
                if self._update_call_count % 300 == 0:
                    print(f"[AgoraSim] Simple collision: {moves_allowed} allowed, {moves_blocked} blocked")
            else:
                # Use simple agent-to-agent collision instead of PhysX raycast
                final_positions = np.copy(proposed_positions)
                agent_radius = 0.6  # Collision radius for agents
                collisions_resolved = 0

                # Simple N^2 collision detection between agents
                for i in range(self._num_agents):
                    for j in range(i + 1, self._num_agents):
                        # Check distance between agents
                        dx = final_positions[i][0] - final_positions[j][0]
                        dy = final_positions[i][1] - final_positions[j][1]
                        distance = np.sqrt(dx*dx + dy*dy)

                        if distance < agent_radius * 2:  # Too close
                            # Push agents apart
                            if distance > 0.001:  # Avoid division by zero
                                # Normalize direction
                                nx = dx / distance
                                ny = dy / distance

                                # Push each agent away by half the overlap
                                overlap = (agent_radius * 2) - distance
                                push = overlap * 0.5

                                final_positions[i][0] += nx * push
                                final_positions[i][1] += ny * push
                                final_positions[j][0] -= nx * push
                                final_positions[j][1] -= ny * push

                                collisions_resolved += 1

                # Wall collision detection (hardcoded U-corral + dynamic rigid body cubes)
                wall_bounces = 0

                # Get dynamic rigid body cube obstacles
                dynamic_obstacles = self._get_rigid_body_obstacles()

                for i in range(self._num_agents):
                    x, y = final_positions[i]

                    # Hardcoded U-corral walls
                    # Bottom wall: y = -10, x = [-20, 20]
                    if -22 <= x <= 22 and y <= -8:
                        final_positions[i][1] = -8
                        wall_bounces += 1

                    # Left wall: x = -20, y = [-10, 10]
                    if x <= -18 and -12 <= y <= 12:
                        final_positions[i][0] = -18
                        wall_bounces += 1

                    # Right wall: x = 20, y = [-10, 10]
                    if x >= 18 and -12 <= y <= 12:
                        final_positions[i][0] = 18
                        wall_bounces += 1

                    # Dynamic rigid body cube obstacles
                    for obstacle in dynamic_obstacles:
                        min_x, max_x, min_y, max_y = obstacle

                        # Check if agent would be inside obstacle bounds
                        if min_x - agent_radius <= x <= max_x + agent_radius and \
                           min_y - agent_radius <= y <= max_y + agent_radius:

                            # Push agent to nearest edge
                            # Calculate distances to each edge
                            dist_left = abs(x - (min_x - agent_radius))
                            dist_right = abs(x - (max_x + agent_radius))
                            dist_bottom = abs(y - (min_y - agent_radius))
                            dist_top = abs(y - (max_y + agent_radius))

                            # Find closest edge and push agent there
                            min_dist = min(dist_left, dist_right, dist_bottom, dist_top)

                            if min_dist == dist_left:
                                final_positions[i][0] = min_x - agent_radius
                            elif min_dist == dist_right:
                                final_positions[i][0] = max_x + agent_radius
                            elif min_dist == dist_bottom:
                                final_positions[i][1] = min_y - agent_radius
                            else:  # min_dist == dist_top
                                final_positions[i][1] = max_y + agent_radius

                            wall_bounces += 1

                if self._update_call_count % 300 == 0:  # Debug every 5 seconds
                    obstacle_count = len(dynamic_obstacles)
                    print(f"[AgoraSim] Simple collision: {collisions_resolved} agent collisions, {wall_bounces} wall bounces, {obstacle_count} dynamic obstacles")

            # Update PointInstancer positions (single bulk update instead of N individual transforms)
            self._update_point_instancer(final_positions)

            # 4) Copy corrected positions back to GPU (reuse buffer)
            tmp = wp.array(final_positions, dtype=wp.vec2, device=self._positions_wp.device)
            wp.copy(self._positions_wp, tmp)

        except Exception as exc:
            print(f"[AgoraSim] ERROR: Collision/check or GPU copy-back failed: {exc}")
            self._update_status("Sim error (collision/update)", "error")
            self._is_running = False
            self._run_btn.text = "Start"
