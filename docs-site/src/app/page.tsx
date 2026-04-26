"use client";

import { useState } from "react";
import Sidebar from "@/components/Sidebar";

export default function Home() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <div className="min-h-screen bg-surface">
      <Sidebar isOpen={isSidebarOpen} setIsOpen={setIsSidebarOpen} />

      <main className="lg:ml-[280px] p-8 lg:p-12 max-w-5xl mx-auto">
        <header className="flex h-16 items-center justify-between gap-4 mb-20 lg:hidden">
          <button
            onClick={() => setIsSidebarOpen(true)}
            className="p-2 -ml-2 text-on-surface hover:text-primary transition-colors"
          >
            <span className="material-symbols-outlined text-2xl">menu</span>
          </button>
          <div className="flex-1">
            <p className="font-headline font-bold text-on-surface text-sm uppercase tracking-tighter">Workflow Docs</p>
          </div>
        </header>

        <section className="mb-24 mt-12 bg-surface text-on-surface border border-surface-container p-12 relative overflow-hidden blueprint-grid">
          <div className="relative z-10">
            <p className="label-md text-primary mb-4">WORKFLOW_DOCUMENTATION // SINGLE_SOURCE_OF_TRUTH</p>
            <h1 className="display-lg mb-8">ComfyUI Workflow Encyclopedia</h1>
            <p className="text-on-surface-variant max-w-2xl leading-relaxed mb-12">
              Every workflow from VNCCS, VNCCS-Utils, HY-Motion, and Pose Director &mdash; explained,
              connected, and documented. Because workflows chain off each other, and one bad output
              ruins everything downstream.
            </p>
            <div className="flex flex-wrap gap-6">
              <a href="/docs" className="bg-on-surface text-surface px-8 py-4 font-headline font-bold uppercase tracking-widest text-[11px] hover:bg-primary transition-all">
                Pipeline Overview
              </a>
              <a href="/docs/pipeline" className="border border-outline-variant px-8 py-4 font-headline font-bold uppercase tracking-widest text-[11px] hover:bg-surface-container transition-all">
                Pipeline Flow
              </a>
            </div>
          </div>
        </section>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
          <div className="p-8 border border-surface-container bg-surface-container-lowest">
            <h3 className="label-md mb-6">26 WORKFLOWS</h3>
            <p className="text-sm text-on-surface-variant leading-relaxed">
              From 4 extensions: VNCCS, VNCCS-Utils, HY-Motion, Pose Director. Each documented with inputs, outputs, and dependencies.
            </p>
          </div>
          <div className="p-8 border border-surface-container bg-surface-container-lowest">
            <h3 className="label-md mb-6">CHAIN DEPENDENCIES</h3>
            <p className="text-sm text-on-surface-variant leading-relaxed">
              Stage A feeds Stage B feeds Stage C. Understand the chain or watch your pipeline break silently.
            </p>
          </div>
          <div className="p-8 border border-surface-container bg-surface-container-lowest">
            <h3 className="label-md mb-6">APP BUILDER</h3>
            <p className="text-sm text-on-surface-variant leading-relaxed">
              All 26 workflows converted to ComfyUI Apps. One sidebar, clean inputs, no node graph hunting.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
