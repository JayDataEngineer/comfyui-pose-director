"use client";

import { useState, useEffect, useRef } from "react";
import Fuse from "fuse.js";
import Link from "next/link";
import { createPortal } from "react-dom";

const docsIndex = [
  { title: "Pipeline Overview", path: "/docs", content: "All 26 workflows across VNCCS, VNCCS-Utils, HY-Motion, Pose Director.", category: "Pipeline" },
  { title: "Pipeline Flow", path: "/docs/pipeline", content: "Data flow between stages: character gen, motion, VNCCS edit, video.", category: "Pipeline" },
  { title: "Z-Image Character Gen", path: "/docs/stages/character-gen", content: "Stage A: generate character portrait with Z-Image Turbo.", category: "Stages" },
  { title: "HY-Motion", path: "/docs/stages/hy-motion", content: "Stage C: text-to-3D motion via trained Hunyuan DiT.", category: "Stages" },
  { title: "MotionDirector", path: "/docs/stages/motion-director", content: "Alternative Stage C: LLM-driven keyframe generation.", category: "Stages" },
  { title: "VNCCS Edit", path: "/docs/stages/vnccs-edit", content: "Stage D: apply pose to character via reference latent injection.", category: "Stages" },
  { title: "LTX Video 2.3", path: "/docs/stages/ltx-video", content: "Stage E: animate between keyframes with LTX Video.", category: "Stages" },
  { title: "Post-Processing", path: "/docs/stages/post-process", content: "Stage F: fix artifacts with LTX EditAnything IC-LoRA.", category: "Stages" },
  { title: "Lip Sync", path: "/docs/stages/lip-sync", content: "Stage G: audio-driven mouth animation (placeholder).", category: "Stages" },
  { title: "TRELLIS 3D", path: "/docs/stages/trellis", content: "Roadmap: image-to-3D mesh generation for better pose reference.", category: "Stages" },
  { title: "VNCCS Char Sheet", path: "/docs/vnccs/char-sheet", content: "Generate multi-angle character sheets.", category: "VNCCS" },
  { title: "VNCCS Clone Character", path: "/docs/vnccs/clone-character", content: "Recreate a character from an existing image.", category: "VNCCS" },
  { title: "VNCCS Clothes", path: "/docs/vnccs/clothes", content: "Change character clothing with reference consistency.", category: "VNCCS" },
  { title: "VNCCS Emotion", path: "/docs/vnccs/emotion", content: "Change facial expressions on existing characters.", category: "VNCCS" },
  { title: "Camera Control", path: "/docs/vnccs-utils/camera", content: "Re-render from different camera angles via Multiple Angles LoRA.", category: "VNCCS-Utils" },
  { title: "Qwen Detailer", path: "/docs/vnccs-utils/detailer", content: "Focused facial expression editing.", category: "VNCCS-Utils" },
  { title: "Pose Studio", path: "/docs/vnccs-utils/pose-studio", content: "Interactive OpenPose skeleton editor.", category: "VNCCS-Utils" },
  { title: "Node Catalog", path: "/docs/reference/nodes", content: "All 8 Pose Director custom nodes.", category: "Reference" },
  { title: "Model Inventory", path: "/docs/reference/models", content: "All models used across the pipeline with sizes.", category: "Reference" },
  { title: "App Builder", path: "/docs/reference/app-builder", content: "Auto-generated .app.json files for ComfyUI Apps.", category: "Reference" },
  { title: "App System Guide", path: "/docs/guides/app-system", content: "How the automated ComfyUI App generation system works end-to-end.", category: "Guides" },
];

export default function Search() {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState(docsIndex);
  const [mounted, setMounted] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { setMounted(true); }, []);

  const fuse = new Fuse(docsIndex, {
    keys: ["title", "content", "category"],
    threshold: 0.3,
  });

  useEffect(() => {
    if (query.length > 0) {
      setResults(fuse.search(query).map((r) => r.item));
    } else {
      setResults(docsIndex);
    }
  }, [query]);

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setIsOpen((open) => !open);
      }
    };
    document.addEventListener("keydown", down);
    return () => document.removeEventListener("keydown", down);
  }, []);

  if (!mounted) return null;

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="w-full lg:max-w-xs flex items-center gap-4 px-4 py-3 bg-surface-container-high border-none transition-all text-xs font-bold font-headline uppercase tracking-widest text-on-surface-variant group hover:bg-surface-container-highest"
      >
        <span className="material-symbols-outlined text-[18px] group-hover:text-primary transition-colors">search</span>
        <span className="flex-1 text-left">Search Workflows...</span>
        <div className="hidden sm:flex items-center gap-1 opacity-40 font-mono text-[9px]">
          CMD+K
        </div>
      </button>

      {isOpen && createPortal(
        <div className="fixed inset-0 z-[100] flex items-start justify-center pt-[15vh] px-4">
          <div
            className="fixed inset-0 bg-on-surface/40 backdrop-blur-md"
            onClick={() => setIsOpen(false)}
          />

          <div className="w-full max-w-2xl bg-white overflow-hidden relative z-10 border-none shadow-[0_40px_80px_rgba(0,104,117,0.12)]">
            <div className="p-6 border-b border-surface-container flex items-center gap-6">
              <span className="material-symbols-outlined text-primary text-[24px]">search</span>
              <input
                ref={inputRef}
                autoFocus
                placeholder="Search workflows, nodes, models..."
                className="flex-1 bg-transparent border-none outline-none text-on-surface text-lg font-headline font-bold uppercase tracking-tight placeholder:text-surface-dim"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
              <button
                onClick={() => setIsOpen(false)}
                className="text-surface-dim hover:text-on-surface transition-colors"
              >
                <span className="material-symbols-outlined">close</span>
              </button>
            </div>

            <div className="max-h-[60vh] overflow-y-auto p-4 space-y-1">
              {results.length > 0 ? (
                results.map((result) => (
                  <Link
                    key={result.path}
                    href={result.path}
                    onClick={() => setIsOpen(false)}
                    className="flex items-center gap-6 p-5 hover:bg-surface-container-low transition-all group"
                  >
                    <div className="w-12 h-12 bg-surface-container flex items-center justify-center text-outline group-hover:bg-primary group-hover:text-on-primary transition-all">
                      <span className="material-symbols-outlined text-[20px]">
                        {result.category === 'Pipeline' ? 'account_tree' :
                         result.category === 'Stages' ? 'layers' :
                         result.category === 'VNCCS' ? 'person' :
                         result.category === 'Reference' ? 'menu_book' :
                         'description'}
                      </span>
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-bold font-headline uppercase tracking-widest text-xs text-on-surface group-hover:text-primary transition-colors">{result.title}</span>
                        <span className="text-[9px] uppercase tracking-[0.2em] text-outline font-mono">{result.category}</span>
                      </div>
                      <p className="text-xs text-on-surface-variant line-clamp-1 opacity-70">{result.content}</p>
                    </div>
                  </Link>
                ))
              ) : (
                <div className="py-20 text-center text-surface-dim text-[10px] tracking-[0.3em] uppercase font-headline font-bold">
                  No matching workflows found.
                </div>
              )}
            </div>
          </div>
        </div>,
        document.body
      )}
    </>
  );
}
