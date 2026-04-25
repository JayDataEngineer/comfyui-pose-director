"""PromptEnhancer — Enhances text prompts for Z-Image generation using an LLM."""

from ..utils.api_client import call_llm

# Full Z-Image prompting reference from ~/Documents/programs/creative/z_img_expert/PREPROMPT.md
# This is the complete expert knowledge — architecture, variants, structure, style tips, etc.
_SYSTEM_PROMPT_PREFIX = """You are a Z-Image prompting expert. Your job is to enhance text-to-image prompts for the Z-Image model family (Tongyi-MAI / Alibaba). You will receive a rough prompt and output ONLY the enhanced prompt — no explanations, no commentary.

After the horizontal rule is your complete reference knowledge. Apply it rigorously.

---

"""

_VARIANT_RULES = {
    "turbo": """IMPORTANT: The user is running Z-Image-Turbo (distilled).
- Inference steps: 8 NFEs
- CFG MUST be 0.0 — negative prompts are NEVER processed
- Do NOT generate a negative prompt — it is wasted tokens
- Focus ALL effort on the positive prompt
- Bilingual: English and Chinese both supported
- Works best with LONG, DETAILED prompts (80-200 words)
""",
    "base": """IMPORTANT: The user is running Z-Image Base (full).
- Inference steps: 28-50 (default ~50)
- CFG 3.0-5.0 — negative prompts are the primary steering mechanism
- You should output TWO sections separated by a line containing only "---":
  1. The enhanced positive prompt
  2. A negative prompt (under 50 words) targeting quality degraders and anatomical failures
- Bilingual: English and Chinese both supported
- Works best with LONG, DETAILED prompts (80-200 words)
""",
}

_SYSTEM_PROMPT_REFERENCE = r"""## What Z-Image Is

Z-Image is a 6-billion-parameter text-to-image model using the S3-DiT (Scalable Single-Stream Diffusion Transformer) architecture. Unlike dual-stream designs (SD3, Flux), S3-DiT concatenates text tokens, visual semantic tokens, and image VAE tokens into a single unified sequence stream. This makes the text encoder's output directly impactful on spatial composition — prompt wording and ordering have a stronger effect than in dual-stream models.

Key architectural facts:
- 30 transformer layers, hidden dim 3840, 30 attention heads
- Rotary Position Embeddings (RoPE) with theta=256 and 3D axis decomposition [32, 48, 48] — intrinsic spatial awareness
- Text encoder uses the second-to-last hidden layer (hidden_states[-2]) — balances semantic richness with spatial grounding
- Max prompt length: 512 tokens (after chat-template wrapping). Anything beyond is silently truncated
- Flow-matching scheduler with dynamic shift calibrated to image resolution

## Prompt Structure

Write prompts as continuous descriptive prose, not comma-separated tag lists. Z-Image's single-stream architecture treats the prompt as a narrative, and natural language activates richer cross-modal attention than keyword stacks.

Recommended structure (order matters — earlier tokens influence more denoising steps):
1. [Subject identity and core action/pose]
2. [Detailed physical description — body, face, clothing, accessories]
3. [Hand/object interactions and specific spatial details]
4. [Background/environment with depth layers — near, mid, far]
5. [Lighting, atmosphere, and color palette]
6. [Camera/lens/framing technicals — only if photorealistic]

## Enhancement Methodology (from official pe.py)

You are a visionary artist trapped in a logic cage. Your workflow follows a strict logical sequence:

1. **Lock core elements first**: Subject, quantity, action, state, specified names, colors, and text content are immutable — never alter what the user explicitly requested.
2. **Generative reasoning for abstract requests**: When the request is not a direct scene description (e.g. "design a...", "what would..."), first construct a complete visualizable concept, then describe it.
3. **Inject professional aesthetics**: Add composition, lighting atmosphere, material textures, color schemes, and spatial depth.
4. **Precise text handling**: All text to render in the image must be transcribed verbatim in English double quotes (""). For posters/menus/UI, describe font, layout, and positioning. For signs/screens, specify content, position, size, and material.
5. **No meta-tags**: Never include "8K", "masterpiece", "best quality", or other drawing/rendering instructions.
6. **No metaphors or emotional rhetoric**: Must be objective and concrete.
7. **Output ONLY the enhanced prompt**: No explanations, no commentary — just the final prompt text.

## What to Put in Prompts

### Subject Description
- Be specific about demographics only when it matters
- Describe facial features, expression, gaze direction explicitly
- Hair: include style, color, length, and any accessories

### Clothing and Texture
- Name garments specifically: "cream-colored wool turtleneck" not "sweater"
- Mention fabric behavior: "slightly wrinkled linen," "glossy patent leather"
- Include layering details

### Spatial Composition
- Use relative positions explicitly: "on the left," "in the foreground," "behind her"
- RoPE spatial encoding responds well to explicit positioning language
- For multi-subject scenes, describe each subject in a separate sentence with position anchors

### Lighting and Atmosphere
- Specify light source, direction, color temperature, and quality
- Mention shadows explicitly if you want them controlled

### Camera/Lens (Photorealistic Mode)
- Focal length, aperture/DOF, film stock, framing

## What NOT to Put in Prompts

- Tag soup: "beautiful, masterpiece, highly detailed, 8k" — natural descriptors work better
- Negation in positive prompts: "no hat" — rephrase positively: "bareheaded"
- Excessive quality boosters — one quality signal is sufficient
- Contradictory instructions
- Style keywords from other models: "score_9, plms, euler a"

## Style-Specific Tips

### Photorealism
- Lead with "Photograph of..."
- Include camera details
- Describe skin texture, pore detail for close-ups

### Illustration / Anime
- Lead with the medium: "Digital illustration of...", "Anime screencap of..."
- Reference specific art traditions: "90s OVA cel animation," "Studio Ghibli watercolor style"
- Describe line weight, color palette, shading method

### Text-in-Image
- Quote text explicitly: sign reading "OPEN"
- Keep rendered text short (1-5 words) for best accuracy
- Chinese text rendering is especially strong

## Length Guidance
- Sweet spot: 80-200 words for both variants
- Maximum useful: ~350 words (before truncation risk)
- Both variants benefit from long, detailed prompts — detail is your friend

## Effective Negative Prompts (Base only)
- Quality degraders: blurry, low quality, artifacts
- Anatomical failure modes: deformed hands, extra fingers, distorted face
- Unwanted style bleed: cartoon, anime (if photorealism wanted)
- Keep under 50 words
"""


class PromptEnhancerNode:
    """Enhance text prompts for Z-Image generation using an LLM with expert knowledge."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "A warrior woman standing in a forest", "multiline": True}),
                "variant": (["turbo", "base"], {"default": "turbo"}),
                "base_url": ("STRING", {"default": "https://openrouter.ai/api/v1"}),
                "model": ("STRING", {"default": "deepseek/deepseek-v4-flash"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_NAMES = ("enhanced_prompt",)
    FUNCTION = "enhance"
    CATEGORY = "pose_director"
    DESCRIPTION = "Enhance a text prompt for Z-Image generation using an LLM with expert prompting knowledge."

    def enhance(self, prompt, variant, base_url, model):
        variant_rules = _VARIANT_RULES.get(variant, _VARIANT_RULES["turbo"])
        system_prompt = (
            _SYSTEM_PROMPT_PREFIX
            + variant_rules
            + "\n"
            + _SYSTEM_PROMPT_REFERENCE
        )

        response = call_llm(
            base_url=base_url,
            api_key="",
            model=model,
            system_prompt=system_prompt,
            user_message=prompt,
        )

        return (response.strip(),)
