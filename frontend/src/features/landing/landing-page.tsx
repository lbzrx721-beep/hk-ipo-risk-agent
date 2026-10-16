import {
  ArrowRight,
  Clock3,
  FileSearch2,
  Network,
  ShieldCheck,
} from "lucide-react";

import { Button } from "@/components/ui/button";

const capabilities = [
  {
    icon: FileSearch2,
    title: "招股书智能解析",
    description: "解析正文与表格，定位关键风险证据。",
  },
  {
    icon: Network,
    title: "多 Agent 协同分析",
    description: "财务、法务、业务与市场角色协同研判。",
  },
  {
    icon: Clock3,
    title: "多时间窗口预警",
    description: "覆盖上市首日、5日、20日与60日风险。",
  },
  {
    icon: ShieldCheck,
    title: "证据追溯与人工复核",
    description: "风险结论可回到原文，并保留复核记录。",
  },
] as const;

function ProspectusVisual() {
  return (
    <figure className="landing-visual landing-hero-media relative mx-auto w-full max-w-[780px]" aria-label="招股书证据分析与多智能体协同示意图">
      <div className="pointer-events-none absolute inset-[10%] rounded-full bg-primary/10 blur-[90px]" />
      <img
        src="/landing-intelligence-visual.png"
        alt="招股书证据分析、多智能体协同与风险信号示意图"
        className="relative z-10 block h-auto w-full object-contain"
      />
    </figure>
  );
}

export function LandingPage({ onEnter }: { onEnter: () => void }) {
  return (
    <div className="landing-shell relative min-h-screen overflow-hidden bg-[#000811] text-foreground">
      <div className="landing-grid pointer-events-none absolute inset-0 opacity-50" />
      <div className="pointer-events-none absolute -left-40 top-20 size-[520px] rounded-full bg-primary/[0.07] blur-[130px]" />
      <div className="pointer-events-none absolute right-0 top-0 size-[600px] rounded-full bg-[#006b9b]/10 blur-[150px]" />

      <header className="relative z-30 mx-auto flex h-24 w-full max-w-[1560px] items-center px-6 lg:px-10">
        <div className="flex items-center gap-4">
          <span className="grid size-11 place-items-center rounded-2xl border border-primary/35 bg-primary/10 text-primary shadow-[0_0_32px_rgba(42,184,236,0.12)]">
            <ShieldCheck className="size-6" />
          </span>
          <div>
            <strong className="block text-base tracking-wide sm:text-lg">港股 IPO 多智能体风险预警系统</strong>
            <span className="text-[10px] uppercase tracking-[0.28em] text-muted-foreground">Agentic IPO Risk Intelligence</span>
          </div>
        </div>
      </header>

      <main className="relative z-10 mx-auto flex min-h-[calc(100vh-6rem)] w-full max-w-[1560px] flex-col px-6 pb-8 lg:px-10">
        <section className="grid flex-1 items-center gap-10 py-8 lg:grid-cols-[0.88fr_1.12fr] lg:py-4">
          <div className="landing-copy max-w-2xl">
            <p className="mb-6 flex items-center gap-3 text-sm font-medium tracking-[0.16em] text-primary/80">
              <span className="h-px w-10 bg-primary/60" />
              面向证券投研的智能风险分析平台
            </p>
            <h1 className="text-[clamp(3rem,5.2vw,5.8rem)] font-semibold leading-[1.08] tracking-[-0.055em]">
              穿透招股书，
              <span className="block text-primary">预见上市风险</span>
            </h1>
            <p className="mt-8 max-w-xl text-base leading-8 text-muted-foreground sm:text-lg">
              从招股书原文证据出发，融合财务、法务、业务与市场多角色协同分析，形成可定位、可复核、可交付的风险预警结论。
            </p>
            <div className="mt-10 flex flex-wrap items-center gap-5">
              <Button
                size="lg"
                className="landing-primary-button h-14 rounded-xl px-8 text-base font-semibold shadow-[0_16px_50px_rgba(37,186,234,0.2)]"
                onClick={onEnter}
              >
                开始使用 <ArrowRight className="ml-2 size-5 transition-transform group-hover/button:translate-x-1" />
              </Button>
              <button
                type="button"
                className="group flex h-12 items-center gap-2 px-3 text-sm text-muted-foreground transition-colors hover:text-foreground"
                onClick={() => document.getElementById("landing-capabilities")?.scrollIntoView({ behavior: "smooth" })}
              >
                了解分析流程 <ArrowRight className="size-4 transition-transform group-hover:translate-x-1" />
              </button>
            </div>
          </div>

          <div className="hidden lg:block">
            <ProspectusVisual />
          </div>
        </section>

        <section id="landing-capabilities" className="grid border-t border-white/10 md:grid-cols-2 xl:grid-cols-4">
          {capabilities.map(({ icon: Icon, title, description }, index) => (
            <div key={title} className={`group flex min-h-32 gap-4 py-7 xl:px-7 ${index > 0 ? "xl:border-l xl:border-white/10" : ""}`}>
              <Icon className="mt-1 size-7 shrink-0 text-primary/90 transition-transform duration-300 group-hover:-translate-y-1" />
              <div>
                <h2 className="font-medium tracking-wide">{title}</h2>
                <p className="mt-2 text-sm leading-6 text-muted-foreground">{description}</p>
              </div>
            </div>
          ))}
        </section>
      </main>
    </div>
  );
}
