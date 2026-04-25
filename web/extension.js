import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI-PoseDirector.Settings",

    settings: [
        {
            id: "PoseDirector.openrouterApiKey",
            name: "OpenRouter API Key",
            type: "text",
            defaultValue: "",
            category: ["Pose Director", "API Keys", "OpenRouter"],
            tooltip: "API key for OpenRouter. Auto-used when base_url points to openrouter.ai.",
        },
        {
            id: "PoseDirector.geminiApiKey",
            name: "Gemini API Key",
            type: "text",
            defaultValue: "",
            category: ["Pose Director", "API Keys", "Gemini"],
            tooltip: "API key for Google Gemini. Auto-used when base_url points to generativelanguage.googleapis.com.",
        },
    ],
});
