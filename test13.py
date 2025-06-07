import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import ollama
import os
import subprocess
import time
import shutil
from collections import Counter

class HabitatNavigationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Navigation GUI")
        self.root.geometry("800x1000")

        # VLM models
        self.models = ["gemma3:4b", "minicpm-v", "bakllava", "llava"]
        self.model_var = tk.StringVar(value=self.models[0])
        self.query_all_var = tk.BooleanVar(value=False)

        # Paths
        self.image_dir = "images"
        os.makedirs(self.image_dir, exist_ok=True)
        self.current_image_path = os.path.join(self.image_dir, "current_image.png")
        self.goal_image_path = os.path.join(self.image_dir, "goal_image.png")
        self.previous_image_path = os.path.join(self.image_dir, "previous_image.png")
        self.command_file = os.path.join(self.image_dir, "command.txt")

        # Simulator process and navigation state
        self.sim_process = None
        self.last_action = None  # Track the last action for course correction
        self.forward_steps_left = 0  # Track remaining forward steps

        # --- Top: Model selection ---
        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=5)
        tk.Label(top_frame, text="Select VLM Model:").grid(row=0, column=0, padx=5)
        self.model_dropdown = ttk.Combobox(
            top_frame,
            textvariable=self.model_var,
            values=self.models,
            state="readonly"
        )
        self.model_dropdown.grid(row=0, column=1, padx=5)
        self.query_all_check = tk.Checkbutton(
            top_frame,
            text="Query all models",
            variable=self.query_all_var
        )
        self.query_all_check.grid(row=0, column=2, padx=5)

        # --- Buttons ---
        self.launch_button = tk.Button(
            self.root,
            text="Launch Simulator",
            command=self.launch_simulator
        )
        self.launch_button.pack(pady=10)

        self.goal_button = tk.Button(
            self.root,
            text="Select Goal Image",
            command=self.select_goal_image
        )
        self.goal_button.pack(pady=5)

        # --- Images Frame ---
        self.images_frame = tk.Frame(self.root)
        self.images_frame.pack(pady=5)
        self.current_canvas = tk.Label(self.images_frame)
        self.current_canvas.grid(row=0, column=0, padx=10)
        self.goal_canvas = tk.Label(self.images_frame)
        self.goal_canvas.grid(row=0, column=1, padx=10)

        # --- Navigation & Analysis ---
        self.start_button = tk.Button(
            self.root,
            text="Start Navigation",
            command=self.start_navigation
        )
        self.start_button.pack(pady=5)

        self.analyze_button = tk.Button(
            self.root,
            text="Analyze and Step",
            command=self.analyze_and_step
        )
        self.analyze_button.pack(pady=5)

        # --- Response Text ---
        self.response_text = tk.Text(
            self.root,
            wrap=tk.WORD,
            height=8
        )
        self.response_text.pack(pady=10, fill=tk.X)

    def launch_simulator(self):
        if self.sim_process and self.sim_process.poll() is None:
            messagebox.showinfo("Info", "Simulator already running.")
            return
        try:
            self.sim_process = subprocess.Popen(["python", "inter_2.py"])
            messagebox.showinfo("Info", "Simulator launched.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not launch simulator: {e}")

    def select_goal_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.png *.jpeg")]
        )
        if not path:
            return
        img = Image.open(path).resize((512, 512))
        img.save(self.goal_image_path)
        display_img = img.resize((256, 256))
        tk_img = ImageTk.PhotoImage(display_img)
        self.goal_canvas.config(image=tk_img)
        self.goal_canvas.image = tk_img

    def fetch_current_image(self):
        start = time.time()
        while not os.path.exists(self.current_image_path):
            if time.time() - start > 10:
                messagebox.showerror("Error", "Current frame not found.")
                return
            time.sleep(0.1)
        img = Image.open(self.current_image_path).resize((256, 256))
        tk_img = ImageTk.PhotoImage(img)
        self.current_canvas.config(image=tk_img)
        self.current_canvas.image = tk_img

    def fetch_current_image_and_enable_button(self):
        self.fetch_current_image()
        self.analyze_button.config(state=tk.NORMAL)

    def start_navigation(self):
        # Send "start_llm" to enter LLM mode
        with open(self.command_file, 'w') as f:
            f.write("start_llm")
        time.sleep(0.5)  # Give simulator time to process the command
        self.fetch_current_image()

    def annotate_image(self, image_path, label, output_path):
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except IOError:
            font = ImageFont.load_default()
        draw.text((10, 10), label, font=font, fill="red")
        img.save(output_path)
        return output_path

    def analyze_and_step(self):
        if self.forward_steps_left > 0:
            messagebox.showinfo("Info", "A forward sequence is already in progress.")
            return

        if not os.path.exists(self.current_image_path) or not os.path.exists(self.goal_image_path):
            messagebox.showerror("Error", "Need both current and goal images.")
            return

        # Save current image as previous before fetching new one
        if os.path.exists(self.current_image_path):
            shutil.copy(self.current_image_path, self.previous_image_path)

        movements = ['forward', 'left', 'right', 'stop']

        # Annotate images
        annotated_current = self.annotate_image(self.current_image_path, "CURRENT", os.path.join(self.image_dir, "temp_current.png"))
        annotated_goal = self.annotate_image(self.goal_image_path, "GOAL", os.path.join(self.image_dir, "temp_goal.png"))
        topdown_map_path = os.path.join(self.image_dir, "topdown_map.png")
        annotated_topdown = self.annotate_image(topdown_map_path, "TOPDOWN MAP", os.path.join(self.image_dir, "temp_topdown.png"))
        images = [annotated_current, annotated_goal, annotated_topdown]
        if os.path.exists(self.previous_image_path):
            annotated_previous = self.annotate_image(self.previous_image_path, "PREVIOUS", os.path.join(self.image_dir, "temp_previous.png"))
            images.insert(0, annotated_previous)

        # Determine image description based on available images
        if len(images) == 4:
            image_desc = (
                "- Image 1: Previous view\n"
                "- Image 2: Current view\n"
                "- Image 3: Goal view\n"
                "- Image 4: Top-down map (shows robot position and orientation)\n"
            )
        else:
            image_desc = (
                "- Image 1: Current view\n"
                "- Image 2: Goal view\n"
                "- Image 3: Top-down map (shows robot position and orientation)\n"
            )

        prompt = (
            "You are a visual navigation expert controlling a robot in a 3D simulator. "
            "Your task is to navigate the robot to a goal location by comparing images and using a top-down map.\n\n"
            f"Images provided:\n{image_desc}\n\n"
            "Additional Information:\n"
            "- The top-down map (last image) shows the robot's position and orientation:\n"
            "  - The red dot labeled 'Robot' marks the robot's current position.\n"
            "  - The arrow extending from the robot indicates the direction the robot is facing.\n"
            "  - The red 'G' marks the goal position.\n"
            "  - The white plane represents areas where the robot can move.\n"
            "  - Black areas are out of bounds for the robot.\n"
            "- The current and goal views include the robot's hand in the center; ignore the hand and focus on the background and other objects.\n\n"
            "Instructions:\n"
            "1. Use the top-down map to determine the robot's current position and the direction it is facing (indicated by the arrow).\n"
            "2. Compare the arrow's direction with the position of the 'G' on the top-down map.\n"
            "3. If the arrow points toward the 'G', choose 'forward'.\n"
            "4. If the arrow does not point toward the 'G' (e.g., it points toward a wall or opposite direction), choose 'left' or 'right' to rotate the robot until the arrow aligns with the 'G'. Prefer the shorter turn (left if the goal is counterclockwise, right if clockwise).\n"
            "5. Compare the background and objects in the current view with the goal view, ignoring the robot's hand.\n"
            "6. If the current view closely matches the goal view and the arrow aligns with the 'G', choose 'stop'.\n"
            "7. If key objects from the goal view are visible in the current view and the top-down map shows the goal ahead of the arrow, choose 'forward'.\n"
            "8. If key objects are not visible, or the top-down map shows the goal to the left or right of the arrow, choose 'left' or 'right' to rotate and align toward the goal.\n"
            "9. Alternate directions if the last rotation was ineffective, using the top-down map to avoid overshooting.\n"
            "10. If key objects were visible in the previous view but not in the current view, choose 'left' or 'right' to reorient, guided by the top-down map.\n"
            "11. Avoid repeating the same incorrect action based on the last action's outcome.\n\n"
            "Choose one action from: forward, left, right, stop.\n"
            "Respond in the format: action|explanation"
        )

        messages = [
            {'role': 'system', 'content': "Provide one of the possible movements."},
            {'role': 'user', 'content': prompt, 'images': images}
        ]

        # Collect responses
        results = []
        if self.query_all_var.get():
            for model in self.models:
                try:
                    resp = ollama.chat(model=model, messages=messages)
                    raw = resp['message']['content']
                except Exception as e:
                    raw = f"forward|Error: {e}"
                parts = raw.strip().split('|', 1)
                action = parts[0].strip().lower()
                expl = parts[1].strip() if len(parts) > 1 else ''
                if action not in movements:
                    action = 'forward'
                    expl = 'defaulted to forward'
                results.append((model, action, expl))
            # Majority vote
            counts = Counter([r[1] for r in results])
            move, _ = counts.most_common(1)[0]
            explanations = [r[2] for r in results if r[1] == move]
            explanation = "; ".join(explanations)
            display = "\n".join([f"{m}: {a}" for m, a, _ in results])
            display += f"\nMajority action: {move}\nExplanation: {explanation}"
        else:
            model = self.model_var.get()
            try:
                resp = ollama.chat(model=model, messages=messages)
                raw = resp['message']['content']
            except Exception as e:
                raw = f"forward|Error: {e}"
            parts = raw.strip().split('|', 1)
            move = parts[0].strip().lower()
            explanation = parts[1].strip() if len(parts) > 1 else ''
            if move not in movements:
                move = 'forward'
                explanation = 'defaulted to forward'
            display = f"{model}: {move}|{explanation}"

        # Update GUI
        self.response_text.delete(1.0, tk.END)
        self.response_text.insert(tk.END, display)

        # Store the chosen action
        self.last_action = move

        if move == "forward":
            self.forward_steps_left = 5
        else:
            self.forward_steps_left = 0

        # Disable the analyze button
        self.analyze_button.config(state=tk.DISABLED)

        # Execute the step
        self.execute_step(move)

    def execute_step(self, action):
        with open(self.command_file, 'w') as f:
            f.write(action)
        if self.forward_steps_left > 0:
            self.forward_steps_left -= 1
            if self.forward_steps_left > 0:
                self.root.after(15, lambda: self.execute_step("forward"))
            else:
                self.root.after(15, self.fetch_current_image_and_enable_button)
        else:
            self.root.after(15, self.fetch_current_image_and_enable_button)

if __name__ == '__main__':
    root = tk.Tk()
    app = HabitatNavigationApp(root)
    root.mainloop()