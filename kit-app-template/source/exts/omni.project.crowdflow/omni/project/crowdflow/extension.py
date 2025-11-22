import omni.ext
import omni.ui as ui
import ctypes

# Style dictionary to make our UI highly visible
# We'll use this to make a bright yellow window with large text
VISIBLE_STYLE = {
    "Window": {
        "background_color": 0xFF_FDE047, # Bright Yellow
        "border_color": 0xFF_FBBF24,
        "border_width": 2,
        "border_radius": 8,
    },
    "Label": {
        "color": 0xFF_1F2937, # Dark Gray text
        "font_size": 24,
        "alignment": ui.Alignment.CENTER
    },
    "Button": {
        "font_size": 16,
        "margin": 10
    }
}

class CrowdFlowExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        # --- 1. LOUD CONSOLE MESSAGE ---
        # This will be very easy to spot in the terminal log
        print("[ext: omni.project.crowdflow] startup")
        try:
            # Native Windows dialog to make the extension visibly obvious on startup
            ctypes.windll.user32.MessageBoxW(0, "CrowdFlow Extension started", "CrowdFlow", 0x40)
        except Exception:
            # Non-Windows hosts or failures should not block startup
            pass

        # --- 2. HIGHLY VISIBLE WINDOW ---
        # We've made the window larger and applied our bright style
        self._window = ui.Window("CrowdFlow Controls", width=500, height=400)
        self._window.frame.style = VISIBLE_STYLE["Window"]

        # This is a flag to check if the simulation is running
        self._is_running = False

        with self._window.frame:
            with ui.VStack(spacing=5):
                # Add a big, noticeable label
                ui.Label("CrowdFlow IS RUNNING", style=VISIBLE_STYLE["Label"])

                # Define what happens when buttons are clicked
                def on_reset():
                    print("Reset button was clicked!")
                    pass

                def on_toggle_run():
                    print("Start/Stop button was clicked!")
                    pass

                # Create the actual buttons
                reset_button = ui.Button("Reset Simulation", clicked_fn=on_reset)
                run_button = ui.Button("Start/Stop", clicked_fn=on_toggle_run)

                # Apply styles to the buttons
                reset_button.style = VISIBLE_STYLE["Button"]
                run_button.style = VISIBLE_STYLE["Button"]

    def on_shutdown(self):
        print("[ext: omni.project.crowdflow] shutdown")
        try:
            if getattr(self, "_window", None) is not None:
                self._window.destroy()
        except Exception:
            pass
