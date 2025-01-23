# Import the required libraries
import os
from dotenv import load_dotenv
import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from transformers.image_utils import load_image
import litserve as ls


# Load the Environment Variables from .env file
load_dotenv()

# Access token for using the model
access_token = os.environ.get("ACCESS_TOKEN")


class PaliGemma2API(ls.LitAPI):
    """
    PaliGemma2API is a subclass of ls.LitAPI that provides an interface to the PaliGemma2 family of models.

    Methods:
        - setup(device): Initializes the model and processor with the specified device.
        - decode_request(request): Decodes the incoming request to extract the inputs.
        - predict(data): Uses the model to generate a caption for the given input image and language.
        - encode_response(output): Encodes the generated response into a JSON format.
    """

    def setup(self, device):
        """
        Sets up the model and processor on the specified device.
        """
        model_id = "google/paligemma2-3b-ft-docci-448"
        self.model = (
            PaliGemmaForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, token=access_token
            )
            .eval()
            .to(device)
        )
        self.processor = PaliGemmaProcessor.from_pretrained(
            model_id, token=access_token
        )

    def decode_request(self, request):
        """
        Decodes the input request to extract the image path and language.
        """
        image_path = request["image_path"]
        language = request.get("language", "en")
        return image_path, language

    def predict(self, data):
        """
        Generates a caption based on the provided image and language.
        """
        # Get the input data
        image_path, language = data
        image = load_image(image_path)

        # Prepare the prompt for the caption generation
        prompt = f"<image>caption {language}"

        # Prepare the input data for the model
        model_inputs = (
            self.processor(text=prompt, images=image, return_tensors="pt")
            .to(torch.bfloat16)
            .to(self.device)
        )
        input_len = model_inputs["input_ids"].shape[-1]

        # Generate the response using the model
        with torch.inference_mode():
            generation = self.model.generate(**model_inputs, max_new_tokens=100)
            return generation[0][input_len:]

    def encode_response(self, output):
        """
        Encodes the given results into a dictionary format.
        """
        caption = self.processor.decode(output, skip_special_tokens=True)
        return {"caption": caption}


if __name__ == "__main__":
    # Create an instance of the PaliGemma2API class and run the server
    api = PaliGemma2API()
    server = ls.LitServer(api, track_requests=True)
    server.run(port=8000)
