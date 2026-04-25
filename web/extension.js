import { app } from "../../scripts/app.js";

const NODE_NAMES = ["MotionDirector", "PoseManipulator", "KeyframeReviewer"];
const API_KEY_WIDGET = "api_key";

app.registerExtension({
    name: "ComfyUI-PoseDirector.ApiKeySettings",

    settings: [
        {
            id: "PoseDirector.openrouterApiKey",
            name: "OpenRouter API Key",
            type: "text",
            defaultValue: "",
            category: ["Pose Director", "API Keys", "OpenRouter"],
            tooltip: "API key for OpenRouter (deepseek, etc.). Used when api_key widget is left empty and base_url points to openrouter.ai.",
        },
        {
            id: "PoseDirector.geminiApiKey",
            name: "Gemini API Key",
            type: "text",
            defaultValue: "",
            category: ["Pose Director", "API Keys", "Gemini"],
            tooltip: "API key for Google Gemini. Used when api_key widget is left empty and base_url points to generativelanguage.googleapis.com.",
        },
        {
            id: "PoseDirector.llamacppUrl",
            name: "Llama.cpp Base URL",
            type: "text",
            defaultValue: "http://127.0.0.1:8001/v1",
            category: ["Pose Director", "LLM Config", "Local"],
            tooltip: "Base URL for local llama.cpp server. No API key needed.",
        },
    ],

    beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (!NODE_NAMES.includes(nodeData.name)) return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = origOnNodeCreated?.apply(this, arguments);

            // Find the api_key widget and mask it as a password field
            const apiKeyWidget = this.widgets?.find(
                (w) => w.name === API_KEY_WIDGET
            );
            if (apiKeyWidget?.inputEl) {
                apiKeyWidget.inputEl.type = "password";
                apiKeyWidget.inputEl.placeholder =
                    "Leave empty to use .env or settings";
            }

            return result;
        };
    },
});
