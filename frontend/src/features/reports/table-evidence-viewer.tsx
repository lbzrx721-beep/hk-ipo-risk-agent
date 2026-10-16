import { Code2, Expand, TableProperties } from "lucide-react";
import { useMemo, useState } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import type { StoredEvidence } from "@/types";

type EvidenceTableRow = {
  values: string[];
  refs: Array<string | null>;
};

export type TableEvidenceReview = "matched" | "row_column_mismatch" | "header_error" | "value_error";

export const tableEvidenceReviewLabel: Record<TableEvidenceReview, string> = {
  matched: "与原图一致",
  row_column_mismatch: "行列错位",
  header_error: "表头错误",
  value_error: "数值错误",
};

export type EvidenceTableModel = {
  tableId: string | null;
  title: string | null;
  segment: string | null;
  unit: string | null;
  currency: string | null;
  period: string | null;
  headers: string[];
  rows: EvidenceTableRow[];
};

const metadataLabels = {
  "表格ID": "tableId",
  "标题": "title",
  "分段": "segment",
  "单位": "unit",
  "币种": "currency",
  "期间": "period",
} as const;

function parseMarkdownRow(line: string): string[] {
  const trimmed = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  const cells: string[] = [];
  let current = "";
  let escaped = false;
  for (const character of trimmed) {
    if (escaped) {
      current += character;
      escaped = false;
    } else if (character === "\\") {
      escaped = true;
    } else if (character === "|") {
      cells.push(current.trim());
      current = "";
    } else {
      current += character;
    }
  }
  cells.push(current.trim());
  return cells;
}

function isMarkdownDivider(line: string): boolean {
  const cells = parseMarkdownRow(line);
  return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function tableCellPosition(ref: string): { row: number; column: number } | null {
  const match = ref.match(/:R(\d+):C(\d+)$/i);
  return match ? { row: Number(match[1]), column: Number(match[2]) } : null;
}

export function parseTableEvidence(evidence: Pick<StoredEvidence, "quote" | "table_id" | "table_cell_refs" | "table_cell_values">): EvidenceTableModel {
  const lines = evidence.quote.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  const metadata: Record<(typeof metadataLabels)[keyof typeof metadataLabels], string | null> = {
    tableId: evidence.table_id ?? null,
    title: null,
    segment: null,
    unit: null,
    currency: null,
    period: null,
  };

  for (const line of lines) {
    for (const [label, key] of Object.entries(metadataLabels) as Array<[keyof typeof metadataLabels, (typeof metadataLabels)[keyof typeof metadataLabels]]>) {
      const prefix = `${label}:`;
      if (line.startsWith(prefix)) metadata[key] = line.slice(prefix.length).trim() || null;
    }
  }

  const headerIndex = lines.findIndex((line, index) => (
    line.startsWith("|") && Boolean(lines[index + 1]) && isMarkdownDivider(lines[index + 1])
  ));
  let headers = headerIndex >= 0 ? parseMarkdownRow(lines[headerIndex]) : [];
  let parsedRows: string[][] = [];
  if (headerIndex >= 0) {
    for (let index = headerIndex + 2; index < lines.length && lines[index].startsWith("|"); index += 1) {
      parsedRows.push(parseMarkdownRow(lines[index]));
    }
  }

  const cells = new Map<string, { ref: string; value: string }>();
  evidence.table_cell_refs.forEach((ref, index) => {
    const position = tableCellPosition(ref);
    if (position) cells.set(`${position.row}:${position.column}`, { ref, value: evidence.table_cell_values[index] ?? "" });
  });
  const positions = [...cells.keys()].map((key) => key.split(":").map(Number));
  const rowCount = Math.max(parsedRows.length, ...positions.map(([row]) => row), 0);
  const columnCount = Math.max(headers.length, ...positions.map(([, column]) => column), 0);
  if (!headers.length && columnCount) {
    headers = Array.from({ length: columnCount }, (_, index) => index === 0 ? "项目" : `第 ${index + 1} 列`);
  }

  const rows = Array.from({ length: rowCount }, (_, rowIndex): EvidenceTableRow => {
    const values = Array.from({ length: columnCount }, (_, columnIndex) => {
      const stored = cells.get(`${rowIndex + 1}:${columnIndex + 1}`);
      return stored?.value ?? parsedRows[rowIndex]?.[columnIndex] ?? "";
    });
    const refs = Array.from({ length: columnCount }, (_, columnIndex) => (
      cells.get(`${rowIndex + 1}:${columnIndex + 1}`)?.ref ?? null
    ));
    return { values, refs };
  });

  return { ...metadata, headers, rows };
}

function StructuredTable({ model }: { model: EvidenceTableModel }) {
  if (!model.headers.length || !model.rows.length) {
    return (
      <Alert>
        <TableProperties />
        <AlertTitle>未能恢复二维表格</AlertTitle>
        <AlertDescription>可以继续查看PDF原图和下方解析详情。</AlertDescription>
      </Alert>
    );
  }
  return (
    <TooltipProvider>
      <div className="max-h-[min(560px,65vh)] overflow-auto">
        <Table className="min-w-[900px]">
          <TableHeader className="sticky top-0 bg-card">
            <TableRow>
              {model.headers.map((header, index) => (
                <TableHead
                  className={cn(
                    "min-w-40 whitespace-normal py-3 align-bottom text-xs leading-5",
                    index === 0 && "sticky left-0 min-w-56 bg-card",
                  )}
                  key={`${header}-${index}`}
                >
                  {header}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {model.rows.map((row, rowIndex) => (
              <TableRow key={`row-${rowIndex}`}>
                {row.values.map((value, columnIndex) => {
                  const ref = row.refs[columnIndex];
                  const content = (
                    <span className={cn(
                      "block min-w-32 whitespace-normal leading-6",
                      columnIndex > 0 && "text-right font-mono tabular-nums",
                      /^\(.*\)$/.test(value) && "text-destructive",
                    )}>{value || "—"}</span>
                  );
                  return (
                    <TableCell
                      className={cn(
                        ref && "bg-primary/6",
                        columnIndex === 0 && "sticky left-0 bg-card font-medium",
                      )}
                      key={`${rowIndex}-${columnIndex}`}
                    >
                      {ref ? (
                        <Tooltip>
                          <TooltipTrigger render={<span className="block cursor-help" />}>{content}</TooltipTrigger>
                          <TooltipContent>{ref}</TooltipContent>
                        </Tooltip>
                      ) : content}
                    </TableCell>
                  );
                })}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </TooltipProvider>
  );
}

function SourceImage({ imageUrl, evidenceId }: { imageUrl?: string; evidenceId: string }) {
  if (!imageUrl) return <p className="p-4 text-sm text-muted-foreground">本条证据没有生成PDF截图。</p>;
  return (
    <div className="flex min-h-[420px] max-h-[min(760px,72vh)] items-start justify-center overflow-auto bg-muted/15 p-4 sm:p-6">
      <img
        className="h-auto w-full max-w-[1100px] rounded-xl bg-white object-contain shadow-sm ring-1 ring-border/50"
        src={imageUrl}
        alt={`证据 ${evidenceId} 原始表格截图`}
        loading="lazy"
      />
    </div>
  );
}

export function TableEvidenceViewer({ evidence, imageUrl, reviewStatus, onReviewStatusChange }: {
  evidence: StoredEvidence;
  imageUrl?: string;
  reviewStatus?: TableEvidenceReview;
  onReviewStatusChange?: (value: TableEvidenceReview | undefined) => void;
}) {
  const [imageOpen, setImageOpen] = useState(false);
  const model = useMemo(
    () => parseTableEvidence(evidence.display_table ?? evidence),
    [evidence],
  );
  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="font-medium">{model.title || "结构化表格证据"}</p>
          {model.period && <p className="mt-1 line-clamp-2 text-xs leading-5 text-muted-foreground">{model.period}</p>}
        </div>
        <div className="flex flex-wrap gap-2">
          {model.currency && <Badge variant="secondary">币种 {model.currency}</Badge>}
          {model.unit && <Badge variant="secondary">单位 {model.unit}</Badge>}
          {model.segment && <Badge variant="outline">分段 {model.segment}</Badge>}
        </div>
      </div>

      <Alert>
        <TableProperties />
        <AlertTitle>结构化表格与PDF原图对照</AlertTitle>
        <AlertDescription>悬停数值可查看内部单元格来源；括号数值按负数突出显示。</AlertDescription>
      </Alert>

      <Card className="overflow-hidden bg-card/70">
        <CardHeader className="border-b border-border/60">
          <CardTitle className="text-sm">结构化数据</CardTitle>
          <CardDescription>完整保留表头、行列关系和原始数值；表格较宽时可横向滚动。</CardDescription>
          <CardAction><Badge variant="outline">{model.rows.length} 行 · {model.headers.length} 列</Badge></CardAction>
        </CardHeader>
        <CardContent className="p-0"><StructuredTable model={model} /></CardContent>
      </Card>

      <Card className="overflow-hidden bg-card/70">
        <CardHeader className="border-b border-border/60">
          <CardTitle className="text-sm">PDF原始表格</CardTitle>
          <CardDescription>在下方大图中核对表头、行列与数值，点击按钮可全屏查看。</CardDescription>
          <CardAction>
            <Button size="sm" variant="outline" disabled={!imageUrl} onClick={() => setImageOpen(true)}>
              <Expand data-icon="inline-start" />放大原图
            </Button>
          </CardAction>
        </CardHeader>
        <CardContent className="p-0"><SourceImage imageUrl={imageUrl} evidenceId={evidence.evidence_id} /></CardContent>
      </Card>

      <Collapsible>
        <CollapsibleTrigger render={<Button size="sm" variant="outline" />}>
          <Code2 data-icon="inline-start" />查看解析详情
        </CollapsibleTrigger>
        <CollapsibleContent>
          <pre className="mt-3 max-h-72 overflow-auto whitespace-pre-wrap rounded-xl border bg-muted/20 p-4 text-xs leading-6 text-muted-foreground">{evidence.quote}</pre>
        </CollapsibleContent>
      </Collapsible>

      {onReviewStatusChange && (
        <div className="flex flex-col gap-2 rounded-xl border bg-muted/10 p-3">
          <div>
            <p className="text-sm font-medium">人工核对表格</p>
            <p className="mt-1 text-xs text-muted-foreground">选择结果会自动写入本条风险的复核意见。</p>
          </div>
          <ToggleGroup
            className="flex-wrap justify-start"
            size="sm"
            variant="outline"
            value={reviewStatus ? [reviewStatus] : []}
            onValueChange={(values) => onReviewStatusChange(values[0] as TableEvidenceReview | undefined)}
          >
            {(Object.entries(tableEvidenceReviewLabel) as Array<[TableEvidenceReview, string]>).map(([value, label]) => (
              <ToggleGroupItem key={value} value={value}>{label}</ToggleGroupItem>
            ))}
          </ToggleGroup>
        </div>
      )}

      <Dialog open={imageOpen} onOpenChange={setImageOpen}>
        <DialogContent className="max-h-[94vh] max-w-[96vw] overflow-auto sm:max-w-[96vw]">
          <DialogHeader>
            <DialogTitle>PDF表格原图</DialogTitle>
            <DialogDescription>{evidence.evidence_id} · 物理页 {evidence.page}</DialogDescription>
          </DialogHeader>
          {imageUrl && <img className="max-h-[82vh] w-full bg-white object-contain" src={imageUrl} alt={`证据 ${evidence.evidence_id} PDF表格原图`} />}
        </DialogContent>
      </Dialog>
    </div>
  );
}
