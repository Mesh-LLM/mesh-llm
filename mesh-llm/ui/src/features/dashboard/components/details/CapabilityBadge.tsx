import type { ReactNode } from "react";

import { Badge } from "../../../../components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "../../../../components/ui/tooltip";

/* eslint-disable jsx-a11y/prefer-button */
export function CapabilityBadge({
  label,
  icon,
  tooltip,
}: {
  label: string;
  icon: ReactNode;
  tooltip?: string;
}) {
  const badge = (
    <Badge className="gap-1.5 rounded-full border-sky-300 bg-sky-50 px-3 py-1.5 text-xs font-medium text-foreground dark:border-sky-900 dark:bg-sky-950/30">
      {icon}
      {label}
    </Badge>
  );
  if (!tooltip) return badge;
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        {/* eslint-disable-next-line jsx-a11y/prefer-button, jsx-a11y/prefer-tag-over-role -- span avoids nested <button> when used inside StatCard */}
        <span
          role="button"
          tabIndex={0}
          className="inline-flex rounded-full bg-transparent p-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          {badge}
        </span>
      </TooltipTrigger>
      <TooltipContent side="bottom" align="center" sideOffset={8}>
        {tooltip}
      </TooltipContent>
    </Tooltip>
  );
}
