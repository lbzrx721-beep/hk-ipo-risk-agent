import { useVirtualizer } from "@tanstack/react-virtual";
import { useCallback, useRef } from "react";

import { cn } from "@/lib/utils";

type VirtualListProps<T> = {
  items: T[];
  estimateSize: number;
  getKey: (item: T) => string;
  renderItem: (item: T, index: number) => React.ReactNode;
  className?: string;
};

export function VirtualList<T>({
  items,
  estimateSize,
  getKey,
  renderItem,
  className,
}: VirtualListProps<T>) {
  const parentRef = useRef<HTMLDivElement>(null);
  const itemKey = useCallback((index: number) => getKey(items[index]), [getKey, items]);
  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => estimateSize,
    getItemKey: itemKey,
    overscan: 6,
  });

  return (
    <div ref={parentRef} className={cn("overflow-auto overscroll-contain", className)}>
      <div className="relative w-full" style={{ height: virtualizer.getTotalSize() }}>
        {virtualizer.getVirtualItems().map((virtualRow) => (
          <div
            key={virtualRow.key}
            data-index={virtualRow.index}
            ref={virtualizer.measureElement}
            className="absolute top-0 left-0 w-full pb-2"
            style={{ transform: `translateY(${virtualRow.start}px)` }}
          >
            {renderItem(items[virtualRow.index], virtualRow.index)}
          </div>
        ))}
      </div>
    </div>
  );
}
