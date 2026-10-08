import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import {
  ChevronLeft,
  ChevronRight,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Image as ImageIcon,
} from 'lucide-react';
import { datasetApi } from '@/services/api/dataset';
import type { Image } from '@/types/dataset';

interface ImageViewerPanelProps {
  image: Image | null;
  onPrevious: () => void;
  onNext: () => void;
  canNavigate: boolean;
  currentIndex: number;
  totalCount: number;
}

export function ImageViewerPanel({
  image,
  onPrevious,
  onNext,
  canNavigate,
  currentIndex,
  totalCount,
}: ImageViewerPanelProps) {
  const [imageLoading, setImageLoading] = useState(false);
  const [imageError, setImageError] = useState(false);
  const [zoom, setZoom] = useState(100);

  const handleImageLoad = () => {
    setImageLoading(false);
    setImageError(false);
  };

  const handleImageError = () => {
    setImageLoading(false);
    setImageError(true);
  };

  const handleZoomIn = () => {
    setZoom((prev) => Math.min(prev + 25, 200));
  };

  const handleZoomOut = () => {
    setZoom((prev) => Math.max(prev - 25, 25));
  };

  const handleResetZoom = () => {
    setZoom(100);
  };

  const getImageUrl = (image: Image) => {
    return datasetApi.getImageUrl(image.path);
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between p-2 border-b">
        <div className="flex items-center gap-2">
          <Button variant="ghost" onClick={onPrevious} disabled={!canNavigate}>
            <ChevronLeft />
          </Button>
          <Button variant="ghost" onClick={onNext} disabled={!canNavigate}>
            <ChevronRight />
          </Button>
          <Separator orientation="vertical" />
          <span className="text-muted-foreground">
            {totalCount > 0 ? `${currentIndex} / ${totalCount}` : '0 / 0'}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <Button variant="ghost" onClick={handleZoomOut} disabled={!image || zoom <= 25}>
            <ZoomOut />
          </Button>
          <span className="text-muted-foreground min-w-12 text-center">{zoom}%</span>
          <Button variant="ghost" onClick={handleZoomIn} disabled={!image || zoom >= 200}>
            <ZoomIn />
          </Button>
          <Button variant="ghost" onClick={handleResetZoom} disabled={!image || zoom === 100}>
            <RotateCcw />
          </Button>
        </div>
      </div>

      <div className="flex-1 min-h-0 p-4">
        {!image ? (
          <div className="flex h-full flex-col items-center justify-center text-center">
            <ImageIcon className="h-24 w-24 text-muted-foreground mb-6" />
            <p className="text-muted-foreground">选择文件查看图片</p>
          </div>
        ) : imageError ? (
          <div className="flex h-full flex-col items-center justify-center text-center">
            <ImageIcon className="h-24 w-24 text-muted-foreground mb-6" />
            <p className="text-muted-foreground mb-4">图片加载失败</p>
            <Button
              variant="outline"
              onClick={() => {
                setImageError(false);
                setImageLoading(true);
              }}
            >
              重试
            </Button>
          </div>
        ) : (
          <div className="relative h-full w-full overflow-auto flex items-center justify-center">
            {imageLoading && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-muted-foreground">加载中...</div>
              </div>
            )}
            <img
              src={getImageUrl(image)}
              alt={image.fileName}
              onLoad={handleImageLoad}
              onError={handleImageError}
              onLoadStart={() => setImageLoading(true)}
            />
          </div>
        )}
      </div>

      {image && (
        <div className="border-t p-4">
          <div className="flex items-center justify-around gap-6 text-sm">
            <div className="flex-1">
              <span className="text-muted-foreground font-medium block mb-1">文件名</span>
              <p className="font-medium truncate" title={image.fileName}>
                {image.fileName}
              </p>
            </div>
            <div className="flex-1">
              <span className="text-muted-foreground font-medium block mb-1">标签</span>
              <p className="font-medium truncate" title={image.label}>
                {image.label || '无标签'}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
