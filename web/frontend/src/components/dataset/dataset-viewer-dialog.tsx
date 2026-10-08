import { useState } from 'react';
import { Dialog, DialogContent } from '@/components/ui/dialog';
import { FileExplorerPanel } from './file-explorer-panel';
import { ImageViewerPanel } from './image-viewer-panel';
import { Separator } from '@/components/ui/separator';
import { DialogDescription, DialogTitle } from '@radix-ui/react-dialog';
import { useImages } from '@/hooks/dataset/use-images';
import { LoadingFallback } from '@/components/loading-fallback';
import type { Image } from '@/types/dataset';

interface DatasetViewerDialogProps {
  datasetId: number;
  open: boolean;
  onOpenChange?: (value: boolean) => void;
}

export function DatasetViewerDialog({ datasetId, open, onOpenChange }: DatasetViewerDialogProps) {
  const { data: images = [], isLoading } = useImages(datasetId);
  const [selectedImage, setSelectedImage] = useState<Image | null>(null);

  const handlePreviousImage = () => {
    if (!selectedImage || images.length === 0) return;
    const currentIndex = images.findIndex((img) => img.id === selectedImage.id);
    const previousIndex = currentIndex > 0 ? currentIndex - 1 : images.length - 1;
    setSelectedImage(images[previousIndex]);
  };

  const handleNextImage = () => {
    if (!selectedImage || images.length === 0) return;
    const currentIndex = images.findIndex((img) => img.id === selectedImage.id);
    const nextIndex = currentIndex < images.length - 1 ? currentIndex + 1 : 0;
    setSelectedImage(images[nextIndex]);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogTitle></DialogTitle>
      <DialogDescription></DialogDescription>
      <DialogContent
        className="p-0 gap-0 max-w-6xl! max-h-[90vh] w-[95vw] h-[85vh]"
        showCloseButton={false}
      >
        {isLoading ? (
          <LoadingFallback />
        ) : (
          <div className="flex flex-1 min-h-0">
            <div className="w-1/4 min-w-[280px]">
              <FileExplorerPanel
                images={images}
                selectedImageId={selectedImage?.id ?? 0}
                onImageSelect={(item) => {
                  setSelectedImage(item);
                }}
              />
            </div>

            <Separator orientation="vertical" />

            <div className="flex-1 flex flex-col min-w-0">
              <ImageViewerPanel
                image={selectedImage}
                onPrevious={handlePreviousImage}
                onNext={handleNextImage}
                canNavigate={images.length > 1}
                currentIndex={
                  selectedImage ? images.findIndex((img) => img.id === selectedImage.id) + 1 : 0
                }
                totalCount={images.length}
              />
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
