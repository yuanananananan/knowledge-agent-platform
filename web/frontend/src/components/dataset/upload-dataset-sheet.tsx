import { useState } from 'react';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
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
import { useUploadDataset } from '@/hooks/dataset/use-upload-dataset';
import type { Modality } from '@/types/dataset';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { getModalityOptions } from '@/lib/modality';

export function UploadDatasetSheet() {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [modality, setModality] = useState<Modality>('SAR');

  const uploadMutation = useUploadDataset();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  const handleSubmit = async () => {
    // 表单验证
    if (!name.trim()) {
      toast.error('请输入数据集名称');
      return;
    }

    if (!file) {
      toast.error('请选择要上传的压缩包');
      return;
    }

    uploadMutation.mutate(
      {
        name: name.trim(),
        file: file,
        modality: modality,
      },
      {
        onSuccess: () => {
          // 重置表单
          setName('');
          setFile(null);
          setModality('SAR');

          // 关闭 Sheet
          setOpen(false);
        },
      }
    );
  };

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button>上传数据集</Button>
      </SheetTrigger>
      <SheetContent>
        <SheetHeader>
          <SheetTitle>上传数据集</SheetTitle>
          <SheetDescription>
            上传数据集压缩包,数据集需按照标准格式组织,上传完成后会自动在数据库中每个类别各添加10张图像
          </SheetDescription>
        </SheetHeader>
        <div className="grid flex-1 auto-rows-min gap-6 px-4">
          <div className="grid gap-3">
            <Label htmlFor="dataset-name">名称</Label>
            <Input
              id="dataset-name"
              placeholder="数据集名称..."
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={uploadMutation.isPending}
            />
          </div>
          <div className="grid gap-3">
            <Label htmlFor="dataset-modality">模态</Label>
            <Select
              value={modality}
              onValueChange={(value: Modality) => {
                setModality(value);
              }}
            >
              <SelectTrigger className="w-full">
                <SelectValue placeholder="模态筛选" />
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
          <div className="grid gap-3">
            <Label htmlFor="dataset-file">压缩包</Label>
            <Input
              id="dataset-file"
              type="file"
              onChange={handleFileChange}
              disabled={uploadMutation.isPending}
            />
          </div>
        </div>
        <SheetFooter>
          <Button type="submit" onClick={handleSubmit} disabled={uploadMutation.isPending}>
            {uploadMutation.isPending ? '上传中...' : '上传'}
          </Button>
          <SheetClose asChild>
            <Button variant="outline" disabled={uploadMutation.isPending}>
              取消
            </Button>
          </SheetClose>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  );
}
