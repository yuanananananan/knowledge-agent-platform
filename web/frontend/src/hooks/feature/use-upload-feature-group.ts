import { useMutation, useQueryClient } from '@tanstack/react-query';
import { featureApi } from '@/services/api/feature';
import { toast } from 'sonner';

export const useUploadFeatureGroup = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: featureApi.uploadFeatureGroup,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['feature-groups'] });
      toast.success('特征组合创建成功');
    },
    onError: (error: Error) => {
      toast.error(`上传失败: ${error.message}`);
    },
  });
};
