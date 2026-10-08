import { useState } from 'react';
import {
  Sheet,
  SheetClose,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '@/components/ui/sheet';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { useCreateFeature } from '@/hooks/feature/use-create-feature';
import type { Modality } from '@/types/dataset';
import { toast } from 'sonner';
import { getModalityOptions } from '@/lib/modality';

export function AddFeatureSheet() {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState('');
  const [modality, setModality] = useState('');
  const [description, setDescription] = useState('');
  const [paramsSchema, setParamsSchema] = useState('');

  const { mutate: createFeature, isPending } = useCreateFeature();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // 基础验证
    if (!name.trim()) {
      toast.error('请输入特征算子名称');
      return;
    }

    if (!modality) {
      toast.error('请选择模态类型');
      return;
    }

    if (!description.trim()) {
      toast.error('请输入算子描述');
      return;
    }

    if (!paramsSchema.trim()) {
      toast.error('请输入参数 schema');
      return;
    }

    // 验证 JSON 格式
    try {
      JSON.parse(paramsSchema);
    } catch {
      toast.error('参数 schema 必须是有效的 JSON 格式');
      return;
    }

    createFeature(
      {
        name: name.trim(),
        modality: modality as Modality,
        description: description.trim(),
        paramsSchema: paramsSchema.trim(),
      },
      {
        onSuccess: () => {
          // 重置表单
          setName('');
          setModality('');
          setDescription('');
          setParamsSchema('');
          setOpen(false);
        },
      }
    );
  };

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button>添加算子</Button>
      </SheetTrigger>
      <SheetContent className="overflow-y-auto">
        <SheetHeader>
          <SheetTitle>添加特征算子</SheetTitle>
          <SheetDescription>填写特征算子的基本信息和参数配置</SheetDescription>
        </SheetHeader>
        <div className="grid flex-1 auto-rows-min gap-6 px-4">
          <div className="space-y-2">
            <Label htmlFor="name">算子名称</Label>
            <Input
              id="name"
              placeholder="例如: extract_mic_mean"
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={isPending}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="modality">模态类型</Label>
            <Select value={modality} onValueChange={setModality} disabled={isPending}>
              <SelectTrigger id="modality" className="w-full">
                <SelectValue placeholder="选择模态类型" />
              </SelectTrigger>
              <SelectContent>
                {getModalityOptions().map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="description">算子描述</Label>
            <Textarea
              id="description"
              placeholder="请输入特征算子的功能描述"
              className="resize-none"
              rows={3}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              disabled={isPending}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="paramsSchema">参数 Schema (JSON)</Label>
            <Textarea
              id="paramsSchema"
              placeholder='{"type": "object", "properties": {...}}'
              className="resize-none font-mono text-sm"
              rows={10}
              value={paramsSchema}
              onChange={(e) => setParamsSchema(e.target.value)}
              disabled={isPending}
            />
            <p className="text-sm text-muted-foreground">请输入符合 JSON Schema 规范的参数配置</p>
          </div>
        </div>
        <SheetFooter>
          <Button type="submit" onClick={handleSubmit} disabled={isPending}>
            {isPending ? '上传中...' : '上传'}
          </Button>
          <SheetClose asChild>
            <Button variant="outline" disabled={isPending}>
              取消
            </Button>
          </SheetClose>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  );
}
