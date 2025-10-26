import sys
import tensorflow as tf

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

# Pre-trained masked language model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

# Constants for generating attention diagrams
GRID_SIZE = 40
PIXELS_PER_WORD = 200


def _load_font():
    """
    Try to load the project font; fall back to PIL default if unavailable.
    """
    try:
        return ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
    except Exception:
        return ImageFont.load_default()


FONT = _load_font()


def main():
    text = input("Text: ")

    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="tf")
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        sys.exit(
            f"Input must include mask token {tokenizer.mask_token} "
            f"(e.g., 'Paris is the {tokenizer.mask_token} of France.')"
        )

    # Use model to process input
    model = TFBertForMaskedLM.from_pretrained(MODEL)
    result = model(**inputs, output_attentions=True)

    # Generate top-K predictions for the [MASK]
    logits_np = safe_numpy(result.logits)
    mask_token_logits = logits_np[0, mask_token_index]
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    for token in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

    # Visualize attentions (align tokens to actual model inputs)
    token_ids = safe_numpy(inputs["input_ids"])[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    visualize_attentions(tokens, result.attentions)


def get_mask_token_index(mask_token_id, inputs):
    """
    Return the index (0-based) of the mask token ID inside the input IDs.
    If not present, return None. Assumes at most one [MASK].
    """
    input_ids = safe_numpy(inputs["input_ids"])[0].tolist()
    try:
        return input_ids.index(mask_token_id)
    except ValueError:
        return None


def get_color_for_attention_score(attention_score):
    """
    Map an attention score in [0, 1] to an RGB gray scale triple.
      0   -> (0, 0, 0)   (black)
      1   -> (255, 255, 255) (white)
    Values in between scale linearly; rounding is acceptable per spec.
    """
    s = float(attention_score)
    if s < 0.0:
        s = 0.0
    elif s > 1.0:
        s = 1.0
    gray = int(round(s * 255))
    return (gray, gray, gray)  # RGB per specification


def visualize_attentions(tokens, attentions):
    """
    Produce attention diagrams for every layer and every head.

    `attentions` is a tuple of tensors (num_layers long).
    Each tensor has shape: (batch=1, num_heads, seq_len, seq_len).
    We generate filenames with 1-indexed layer/head numbers.
    """
    for layer_idx, layer_tensor in enumerate(attentions, start=1):
        # shape: (1, num_heads, seq_len, seq_len)
        layer_np = safe_numpy(layer_tensor)
        heads_np = layer_np[0]  # (num_heads, seq_len, seq_len)
        num_heads = heads_np.shape[0]
        for head_idx in range(1, num_heads + 1):
            weights = heads_np[head_idx - 1]  # (seq_len, seq_len)
            generate_diagram(layer_idx, head_idx, tokens, weights)


def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Render a single attention head diagram as an image.
    Lighter cells = higher attention scores.
    """
    seq_len = len(tokens)
    image_size = GRID_SIZE * seq_len + PIXELS_PER_WORD

    # Use RGBA canvas for clean overlay of rotated text; draw colors in RGB tuples.
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    # Draw token labels for rows and columns
    for i, token in enumerate(tokens):
        # Column labels (rotated)
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )
        token_image = token_image.rotate(90)
        img.paste(token_image, mask=token_image)

        # Row labels (left side)
        try:
            _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        except Exception:
            # Fallback for older Pillow
            width = draw.textlength(token, font=FONT)
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )

    # Draw attention grid
    for i in range(seq_len):
        y = PIXELS_PER_WORD + i * GRID_SIZE
        for j in range(seq_len):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i][j])
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    # Save image
    img.save(f"Attention_Layer{layer_number}_Head{head_number}.png")


def safe_numpy(t):
    """
    Convert a TensorFlow tensor (or numpy array) to numpy array safely.
    """
    if isinstance(t, tf.Tensor):
        try:
            return t.numpy()
        except Exception:
            return tf.convert_to_tensor(t).numpy()
    return t


if __name__ == "__main__":
    main()
