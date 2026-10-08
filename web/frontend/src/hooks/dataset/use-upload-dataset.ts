import { useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { datasetApi } from '@/services/api/dataset';
import { datasetKeys } from './query-keys';
import type { Modality } from '@/types/dataset';

export function useUploadDataset() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: { name: string; file: File; modality: Modality }) =>
      datasetApi.uploadDataset(data),
    onSuccess: () => {
      // Invalidate and refetch dataset list
      queryClient.invalidateQueries({ queryKey: datasetKeys.list() });
      toast.success('数据集上传成功');
    },
    onError: (_error) => {
      toast.error('上传失败，请稍后重试');
    },
  });
}
