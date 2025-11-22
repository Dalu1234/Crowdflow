import omni.ext
import omni.ui as ui

class CrowdFlowExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        self._window = ui.Window("CrowdFlow Controls", width=300, height=200)

        # This is a flag to check if the simulation is running
        self._is_running = False

        with self._window.frame:
            with ui.VStack(spacing=5):

                # Define what happens when buttons are clicked
                def on_reset():
                    print("Reset button was clicked!")
                    pass

                def on_toggle_run():
                    print("Start/Stop button was clicked!")
                    pass

                # Create the actual buttons
                ui.Button("Reset Simulation", clicked_fn=on_reset)
                ui.Button("Start/Stop", clicked_fn=on_toggle_run)

    def on_shutdown(self):
        pass