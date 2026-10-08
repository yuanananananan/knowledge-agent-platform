import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import type { FeatureGroupWithDetails } from '@/types/feature';

interface FeatureGroupDetailsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  group: FeatureGroupWithDetails | null;
}

export function FeatureGroupDetailsDialog({
  open,
  onOpenChange,
  group,
}: FeatureGroupDetailsDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{group?.name || '特征组合详情'}</DialogTitle>
          <DialogDescription>该组合包含 {group?.features.length || 0} 个特征</DialogDescription>
        </DialogHeader>
        {group ? (
          <pre className="text-xs overflow-auto">{JSON.stringify(group, null, 2)}</pre>
        ) : null}
      </DialogContent>
    </Dialog>
  );
}
