import { useState, useMemo } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useCreateFeatureGroup } from '@/hooks/feature/use-create-feature-group';
import type { Feature } from '@/types/feature';
import { toast } from 'sonner';

interface CreateFeatureGroupDialogProps {
  open: boolean;
  onOpenChange: (_open: boolean) => void;
  features: Feature[];
  onSuccess?: () => void;
}

export function CreateFeatureGroupDialog({
  open,
  onOpenChange,
  features,
  onSuccess,
}: CreateFeatureGroupDialogProps) {
  const [name, setName] = useState('');
  const { mutate: createFeatureGroup, isPending } = useCreateFeatureGroup();

  // 检查所有 feature 是否属于同一个 modality
  const modalityValidation = useMemo(() => {
    if (features.length === 0) {
      return { isValid: false, modality: null, error: '请至少选择一个特征' };
    }

    const firstModality = features[0].modality;
    const allSameModality = features.every((f) => f.modality === firstModality);

    if (!allSameModality) {
      return {
        isValid: false,
        modality: null,
        error: '所选特征必须属于同一个模态',
      };
    }

    return { isValid: true, modality: firstModality, error: null };
  }, [features]);

  const handleCreate = () => {
    if (!name.trim()) {
      toast.error('请输入组合名称');
      return;
    }

    if (!modalityValidation.isValid) {
      toast.error(modalityValidation.error);
      return;
    }

    const featureIds = features.map((f) => f.id);
    createFeatureGroup(
      { name: name.trim(), features: featureIds },
      {
        onSuccess: () => {
          onOpenChange(false);
          setName('');
          onSuccess?.();
        },
      }
    );
  };

  const handleOpenChange = (newOpen: boolean) => {
    if (!isPending) {
      onOpenChange(newOpen);
      if (!newOpen) {
        setName('');
      }
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>创建特征组合</DialogTitle>
          <DialogDescription>
            将选中的 {features.length} 个特征算子创建为一个特征组合
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <div className="grid gap-2">
            <Label htmlFor="group-name">组合名称</Label>
            <Input
              id="group-name"
              placeholder="请输入特征组合名称"
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={isPending}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleCreate();
                }
              }}
            />
          </div>

          {/* Modality 验证信息 */}
          {modalityValidation.isValid ? (
            <div className="rounded-md bg-green-50 dark:bg-green-950 p-3 border border-green-200 dark:border-green-800">
              <p className="text-sm text-green-800 dark:text-green-200">
                模态类型: <span className="font-semibold">{modalityValidation.modality}</span>
              </p>
            </div>
          ) : (
            <div className="rounded-md bg-destructive/10 p-3 border border-destructive/30">
              <p className="text-sm text-destructive">{modalityValidation.error}</p>
            </div>
          )}

          <div className="grid gap-2">
            <Label>选中的特征算子</Label>
            <div className="rounded-md border p-3 max-h-40 overflow-y-auto">
              <ul className="text-sm space-y-1">
                {features.map((feature) => (
                  <li key={feature.id} className="flex justify-between text-muted-foreground">
                    <span>{feature.name}</span>
                    <span className="text-xs text-muted-foreground/70">{feature.modality}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => handleOpenChange(false)} disabled={isPending}>
            取消
          </Button>
          <Button
            onClick={handleCreate}
            disabled={isPending || !name.trim() || !modalityValidation.isValid}
          >
            {isPending ? '创建中...' : '创建'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
