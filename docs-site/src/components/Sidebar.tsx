"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navGroups = [
  {
    title: "Pipeline",
    items: [
      { name: "Overview", path: "/docs", icon: "home" },
      { name: "Pipeline Flow", path: "/docs/pipeline", icon: "account_tree" },
    ],
  },
  {
    title: "Character Generation",
    items: [
      { name: "Z-Image Turbo", path: "/docs/stages/character-gen", icon: "person_add" },
      { name: "VNCCS Char Sheet", path: "/docs/vnccs/char-sheet", icon: "grid_view" },
      { name: "VNCCS Clone Char", path: "/docs/vnccs/clone-character", icon: "content_copy" },
    ],
  },
  {
    title: "Character Editing",
    items: [
      { name: "VNCCS Clothes", path: "/docs/vnccs/clothes", icon: "checkroom" },
      { name: "VNCCS Emotion", path: "/docs/vnccs/emotion", icon: "mood" },
      { name: "VNCCS Detailer", path: "/docs/vnccs-utils/detailer", icon: "auto_fix_high" },
      { name: "Camera Control", path: "/docs/vnccs-utils/camera", icon: "videocam" },
      { name: "VNCCS Edit (ours)", path: "/docs/stages/vnccs-edit", icon: "draw" },
    ],
  },
  {
    title: "Pose & Motion",
    items: [
      { name: "HY-Motion", path: "/docs/stages/hy-motion", icon: "directions_run" },
      { name: "Pose Studio", path: "/docs/vnccs-utils/pose-studio", icon: "accessibility_new" },
      { name: "Motion Director", path: "/docs/stages/motion-director", icon: "animation" },
    ],
  },
  {
    title: "Video & Post",
    items: [
      { name: "LTX Video 2.3", path: "/docs/stages/ltx-video", icon: "movie" },
      { name: "Post-Processing", path: "/docs/stages/post-process", icon: "auto_awesome" },
      { name: "Lip Sync", path: "/docs/stages/lip-sync", icon: "mic" },
      { name: "TRELLIS 3D", path: "/docs/stages/trellis", icon: "view_in_ar" },
    ],
  },
  {
    title: "Reference",
    items: [
      { name: "Node Catalog", path: "/docs/reference/nodes", icon: "category" },
      { name: "Model Inventory", path: "/docs/reference/models", icon: "inventory_2" },
      { name: "App Builder", path: "/docs/reference/app-builder", icon: "widgets" },
    ],
  },
  {
    title: "Guides",
    items: [
      { name: "App System", path: "/docs/guides/app-system", icon: "smart_toy" },
    ],
  },
];

export default function Sidebar({ isOpen, setIsOpen }: { isOpen: boolean; setIsOpen: (val: boolean) => void }) {
  const pathname = usePathname();

  return (
    <>
      {/* Mobile Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-on-surface/20 backdrop-blur-sm z-40 lg:hidden"
          onClick={() => setIsOpen(false)}
        />
      )}

      <aside
        className={`
          fixed inset-y-0 left-0 w-[280px] bg-white border-r border-outline-variant/10 z-50
          transition-transform duration-300 ease-in-out
          lg:translate-x-0 ${isOpen ? 'translate-x-0' : '-translate-x-full'}
          overflow-y-auto no-scrollbar
        `}
      >
        <div className="p-8 border-b-0 flex flex-col min-h-full">
          {/* Profile Section */}
          <div className="flex items-center gap-4 mb-12 px-2">
            <div className="w-12 h-12 bg-surface-container-low flex items-center justify-center overflow-hidden">
              <span className="material-symbols-outlined text-primary text-2xl">smart_toy</span>
            </div>
            <div>
              <p className="font-headline font-bold text-on-surface leading-none text-sm uppercase tracking-tighter">Pose Director</p>
              <p className="text-[9px] text-outline font-bold uppercase tracking-[0.2em] mt-1.5 opacity-60">WORKFLOW_DOCS_V1</p>
            </div>
          </div>

          <nav className="space-y-10 flex-1">
            {navGroups.map((group) => (
              <div key={group.title}>
                <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-outline mb-6 px-2 opacity-50 font-headline">
                  {group.title}
                </h3>
                <div className="space-y-1">
                  {group.items.map((item) => {
                    const isActive = pathname === item.path;
                    return (
                      <Link
                        key={item.name}
                        href={item.path}
                        onClick={() => setIsOpen(false)}
                        className={`
                          flex items-center gap-4 px-3 py-3 text-[11px] font-bold font-headline uppercase tracking-[0.15em] transition-all duration-300
                          ${isActive
                            ? 'bg-primary/5 text-primary border-l-4 border-primary'
                            : 'text-on-surface-variant hover:text-primary hover:bg-surface-container-low'}
                        `}
                      >
                        <span className="material-symbols-outlined text-[18px]">
                          {item.icon}
                        </span>
                        {item.name}
                      </Link>
                    );
                  })}
                </div>
              </div>
            ))}
          </nav>

          <div className="mt-12 pt-12 border-t border-outline-variant/10">
            <a href="http://localhost:8465" target="_blank" className="w-full bg-on-surface text-surface text-[10px] font-bold py-4 flex items-center justify-center gap-3 hover:bg-primary transition-colors uppercase tracking-[0.2em] font-headline">
              <span className="material-symbols-outlined text-sm">open_in_new</span>
              Open ComfyUI
            </a>
          </div>
        </div>
      </aside>
    </>
  );
}
