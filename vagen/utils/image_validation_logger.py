import dataclasses

@dataclasses.dataclass
class ValidationGenerationsLogger:
    project_name: str = None
    experiment_name: str = None

    def log(self, loggers, samples, step):
        if "wandb" in loggers:
            self.log_generations_to_wandb(samples, step)
        if "swanlab" in loggers:
            self.log_generations_to_swanlab(samples, step)
        if "mlflow" in loggers:
            self.log_generations_to_mlflow(samples, step)

        if "clearml" in loggers:
            self.log_generations_to_clearml(samples, step)
        if "tensorboard" in loggers:
            self.log_generations_to_tensorboard(samples, step)

        if "vemlp_wandb" in loggers:
            self.log_generations_to_vemlp_wandb(samples, step)

    def log_generations_to_vemlp_wandb(self, samples, step):
        from volcengine_ml_platform import wandb as vemlp_wandb

        self._log_generations_to_wandb(samples, step, vemlp_wandb)

    def log_generations_to_wandb(self, samples, step):
        import wandb

        self._log_generations_to_wandb(samples, step, wandb)

    def _log_generations_to_wandb(self, samples, step, wandb):
        """Log samples to wandb as a table with image support"""
       

        # Check if samples include images (4 elements) or not (3 elements)
        has_images = len(samples[0]) == 4 if samples else False

        # Create column names for all samples
        if has_images:
            columns = ["step"] + sum([[f"input_{i+1}", f"output_{i+1}", f"score_{i+1}", f"image_{i+1}"] for i in range(len(samples))], [])
        else:
            columns = ["step"] + sum([[f"input_{i+1}", f"output_{i+1}", f"score_{i+1}"] for i in range(len(samples))], [])

        if not hasattr(self, 'validation_table'):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(step)
        for sample in samples:
            if has_images:
                input_text, output_text, score, image = sample
                row_data.extend([input_text, output_text, score])
                # Convert PIL Image to wandb.Image if available, otherwise add None
                if image is not None:
                    # Handle list of images (multiple images per sample)
                    if isinstance(image, list):
                        if len(image) == 0:
                            wandb_image = None
                        elif len(image) == 1:
                            wandb_image = wandb.Image(image[0])
                        else:
                            # Concatenate multiple images into a grid
                            import math
                            from PIL import Image

                            num_images = len(image)
                            # Create a grid: calculate rows and columns
                            cols = min(3, num_images)  # Max 3 columns
                            rows = math.ceil(num_images / cols)

                            # Get max dimensions
                            max_width = max(img.width for img in image)
                            max_height = max(img.height for img in image)

                            # Create composite image
                            composite_width = max_width * cols
                            composite_height = max_height * rows
                            composite = Image.new('RGB', (composite_width, composite_height), (255, 255, 255))

                            # Paste images into grid
                            for idx, img in enumerate(image):
                                row = idx // cols
                                col = idx % cols
                                x = col * max_width
                                y = row * max_height
                                composite.paste(img, (x, y))

                            wandb_image = wandb.Image(composite)
                    else:
                        wandb_image = wandb.Image(image)
                    row_data.append(wandb_image)
                else:
                    row_data.append(None)
            else:
                row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({"val/generations": new_table}, step=step)
        self.validation_table = new_table

    def log_generations_to_swanlab(self, samples, step):
        """Log samples to swanlab as text with image support"""
        import swanlab

        # Check if samples include images (4 elements) or not (3 elements)
        has_images = len(samples[0]) == 4 if samples else False

        swanlab_text_list = []
        swanlab_image_list = []

        for i, sample in enumerate(samples):
            if has_images:
                input_text, output_text, score, image = sample
                row_text = f"""
            input: {input_text}

            ---

            output: {output_text}

            ---

            score: {score}
            """
                swanlab_text_list.append(swanlab.Text(row_text, caption=f"sample {i+1}"))

                # Add image if available
                if image is not None:
                    # Handle list of images (multiple images per sample)
                    if isinstance(image, list):
                        if len(image) == 1:
                            swanlab_image_list.append(swanlab.Image(image[0], caption=f"sample {i+1}"))
                        elif len(image) > 1:
                            # Create composite image for multiple images
                            import math
                            from PIL import Image

                            num_images = len(image)
                            cols = min(3, num_images)
                            rows = math.ceil(num_images / cols)

                            max_width = max(img.width for img in image)
                            max_height = max(img.height for img in image)

                            composite_width = max_width * cols
                            composite_height = max_height * rows
                            composite = Image.new('RGB', (composite_width, composite_height), (255, 255, 255))

                            for idx, img in enumerate(image):
                                row = idx // cols
                                col = idx % cols
                                x = col * max_width
                                y = row * max_height
                                composite.paste(img, (x, y))

                            swanlab_image_list.append(swanlab.Image(composite, caption=f"sample {i+1}"))
                    else:
                        swanlab_image_list.append(swanlab.Image(image, caption=f"sample {i+1}"))
            else:
                row_text = f"""
            input: {sample[0]}

            ---

            output: {sample[1]}

            ---

            score: {sample[2]}
            """
                swanlab_text_list.append(swanlab.Text(row_text, caption=f"sample {i+1}"))

        # Log to swanlab
        log_dict = {"val/generations": swanlab_text_list}
        if has_images and swanlab_image_list:
            log_dict["val/generation_images"] = swanlab_image_list
        swanlab.log(log_dict, step=step)


    def log_generations_to_mlflow(self, samples, step):
        """Log validation generation to mlflow as artifacts"""
        # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html?highlight=log_artifact#mlflow.log_artifact

        import json
        import tempfile

        import mlflow

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                validation_gen_step_file = Path(tmp_dir, f"val_step{step}.json")
                row_data = []
                for sample in samples:
                    data = {"input": sample[0], "output": sample[1], "score": sample[2]}
                    row_data.append(data)
                with open(validation_gen_step_file, "w") as file:
                    json.dump(row_data, file)
                mlflow.log_artifact(validation_gen_step_file)
        except Exception as e:
            print(f"WARNING: save validation generation file to mlflow failed with error {e}")

    def log_generations_to_clearml(self, samples, step):
        """Log validation generation to clearml as table"""

        import clearml
        import pandas as pd

        task: clearml.Task | None = clearml.Task.current_task()
        if task is None:
            return

        table = [
            {
                "step": step,
                "input": sample[0],
                "output": sample[1],
                "score": sample[2],
            }
            for sample in samples
        ]

        logger = task.get_logger()
        logger.report_table(
            series="Validation generations",
            title="Validation",
            table_plot=pd.DataFrame.from_records(table),
            iteration=step,
        )

    def log_generations_to_tensorboard(self, samples, step):
        """Log samples to tensorboard as text"""
        # Initialize tensorboard writer if not exists
        if not hasattr(self, "writer"):
            from torch.utils.tensorboard import SummaryWriter

            # Use the same directory structure as _TensorboardAdapter
            if self.project_name and self.experiment_name:
                default_dir = os.path.join("tensorboard_log", self.project_name, self.experiment_name)
            else:
                default_dir = "tensorboard_log"

            tensorboard_dir = os.environ.get("TENSORBOARD_DIR", default_dir)
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)

        # Format the samples data into readable text
        text_content = f"**Generation Results - Step {step}**\n\n"

        for i, sample in enumerate(samples):
            text_content += f"### Sample {i + 1}\n"

            # Assuming sample contains [input, output, score]
            if len(sample) >= 3:
                input_text, output_text, score = sample[0], sample[1], sample[2]

                text_content += f"**Input:** {input_text}\n\n"
                text_content += f"**Output:** {output_text}\n\n"
                text_content += f"**Score:** {score}\n\n"
            else:
                # Handle cases where sample format might be different
                text_content += f"**Data:** {sample}\n\n"

            text_content += "---\n\n"

        # Log to tensorboard as text
        self.writer.add_text("val/generations", text_content, step)
        # Flush to ensure data is written
        self.writer.flush()
