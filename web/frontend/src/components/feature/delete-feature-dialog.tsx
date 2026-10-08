import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { useDeleteFeature } from '@/hooks/feature/use-delete-feature';
import { useDeleteFeatures } from '@/hooks/feature/use-delete-features';
import type { Feature } from '@/types/feature';

interface DeleteFeatureDialogProps {
  open: boolean;
  onOpenChange: (_open: boolean) => void;
  feature: Feature | null;
}

export function DeleteFeatureDialog({ open, onOpenChange, feature }: DeleteFeatureDialogProps) {
  const { mutate: deleteFeature, isPending } = useDeleteFeature();

  const handleDelete = async () => {
    if (!feature) return;

    deleteFeature(feature.id, {
      onSuccess: () => {
        onOpenChange(false);
      },
    });
  };

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle className="text-base font-semibold">删除特征算子</AlertDialogTitle>
          <AlertDialogDescription>
            确定要删除特征算子 <strong>{feature?.name}</strong> 吗？此操作无法撤销。
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel disabled={isPending}>取消</AlertDialogCancel>
          <AlertDialogAction onClick={handleDelete} disabled={isPending}>
            {isPending ? '删除中...' : '确认删除'}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

// 批量删除对话框
interface DeleteFeaturesDialogProps {
  open: boolean;
  onOpenChange: (_open: boolean) => void;
  features: Feature[];
  onSuccess?: () => void;
}

export function DeleteFeaturesDialog({
  open,
  onOpenChange,
  features,
  onSuccess,
}: DeleteFeaturesDialogProps) {
  const { mutate: deleteFeatures, isPending } = useDeleteFeatures();

  const handleDelete = async () => {
    if (features.length === 0) return;

    const featureIds = features.map((f) => f.id);
    deleteFeatures(featureIds, {
      onSuccess: () => {
        onOpenChange(false);
        onSuccess?.();
      },
    });
  };

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle className="text-base font-semibold">批量删除特征算子</AlertDialogTitle>
          <AlertDialogDescription>
            确定要删除选中的 <strong>{features.length}</strong> 个特征算子吗？此操作无法撤销。
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel disabled={isPending}>取消</AlertDialogCancel>
          <AlertDialogAction onClick={handleDelete} disabled={isPending || features.length === 0}>
            {isPending ? '删除中...' : '确认删除'}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
