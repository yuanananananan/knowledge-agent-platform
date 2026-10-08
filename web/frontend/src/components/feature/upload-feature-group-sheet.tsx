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
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { useUploadFeatureGroup } from '@/hooks/feature/use-upload-feature-group';
import { toast } from 'sonner';
import { Input } from '../ui/input';

export function UploadFeatureGroupSheet() {
  const [open, setOpen] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const { mutate: uploadFeatureGroup, isPending } = useUploadFeatureGroup();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      if (selectedFile.type !== 'application/json') {
        toast.error('请上传JSON文件');
        return;
      }
      setFile(selectedFile);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!file) {
      toast.error('请选择文件');
      return;
    }

    try {
      const content = await file.text();
      const data = JSON.parse(content);

      // 验证数据格式
      if (!data.name || !data.description || !data.features || !Array.isArray(data.features)) {
        toast.error('JSON格式不正确，请确保包含所需字段');
        return;
      }

      uploadFeatureGroup(data, {
        onSuccess: () => {
          setFile(null);
          setOpen(false);
        },
      });
    } catch (error) {
      if (error instanceof SyntaxError) {
        toast.error('JSON格式错误，请检查文件内容');
      } else {
        toast.error('上传失败');
      }
    }
  };

  const handleOpenChange = (newOpen: boolean) => {
    setOpen(newOpen);
    if (!newOpen) {
      setFile(null);
    }
  };

  return (
    <Sheet open={open} onOpenChange={handleOpenChange}>
      <SheetTrigger asChild>
        <Button>上传特征组合</Button>
      </SheetTrigger>
      <SheetContent className="overflow-y-auto">
        <SheetHeader>
          <SheetTitle>上传特征组合</SheetTitle>
          <SheetDescription>上传包含特征定义的JSON文件来创建新的特征组合</SheetDescription>
        </SheetHeader>
        <div className="grid flex-1 auto-rows-min gap-6 px-4">
          <div className="space-y-2">
            <Label htmlFor="file">选择JSON文件</Label>
            <Input
              id="file"
              type="file"
              accept=".json"
              onChange={handleFileChange}
              disabled={isPending}
            />
          </div>

          <div className="space-y-2">
            <Label>JSON格式示例</Label>
            <div className="rounded-md bg-muted p-4">
              <pre className="text-xs overflow-auto">
                {JSON.stringify(
                  {
                    name: '测试特征组合',
                    description: '包含测试特征的组合',
                    features: [
                      {
                        name: 'get_test_feat1',
                        description: '测试特征1',
                        modality: 'SAR',
                        code: "print('hello world')",
                      },
                    ],
                  },
                  null,
                  2
                )}
              </pre>
            </div>
            <p className="text-sm text-muted-foreground">
              请确保JSON文件包含name、description和features字段
            </p>
          </div>
        </div>
        <SheetFooter>
          <Button type="submit" onClick={handleSubmit} disabled={!file || isPending}>
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
