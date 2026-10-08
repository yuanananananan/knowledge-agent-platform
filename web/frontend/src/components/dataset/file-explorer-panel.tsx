import { useState, useMemo } from 'react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Input } from '@/components/ui/input';
import { Image as ImageIcon } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { Image } from '@/types/dataset';

interface FileExplorerPanelProps {
  images: Image[];
  selectedImageId: number | null;
  onImageSelect: (image: Image) => void;
}

export function FileExplorerPanel({
  images,
  selectedImageId,
  onImageSelect,
}: FileExplorerPanelProps) {
  const [searchQuery, setSearchQuery] = useState('');

  const filteredImages = useMemo(() => {
    if (!searchQuery.trim()) return images;

    return images.filter((image) =>
      image.fileName.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [images, searchQuery]);

  return (
    <div className="flex flex-col h-full">
      <div className="p-4">
        <Input
          placeholder="搜索文件..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
      </div>

      <ScrollArea className="flex-1">
        {filteredImages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-32 text-center px-4">
            <ImageIcon className="h-8 w-8 text-muted-foreground mb-2" />
            <p className="text-sm text-muted-foreground">
              {searchQuery ? '未找到匹配的文件' : '暂无文件'}
            </p>
          </div>
        ) : (
          <div>
            {filteredImages.map((image) => (
              <div
                key={image.id}
                onClick={() => onImageSelect(image)}
                className={cn(
                  'flex items-center gap-1 p-3 cursor-pointer transition-colors hover:bg-muted',
                  selectedImageId === image.id && 'bg-secondary'
                )}
              >
                <div className="shrink-0">
                  <ImageIcon className="h-4 w-4" />
                </div>
                <p
                  className={cn(
                    'text-sm truncate',
                    selectedImageId === image.id ? 'font-medium' : 'font-normal'
                  )}
                  title={image.fileName}
                >
                  {image.fileName}
                </p>
              </div>
            ))}
          </div>
        )}
      </ScrollArea>
    </div>
  );
}
