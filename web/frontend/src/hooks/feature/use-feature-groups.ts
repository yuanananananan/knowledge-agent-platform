import { useQuery } from '@tanstack/react-query';
import { featureApi } from '@/services/api/feature';
import type { FeatureGroupWithDetails } from '@/types/feature';

export const useFeatureGroups = () => {
  return useQuery({
    queryKey: ['feature-groups'],
    queryFn: async (): Promise<FeatureGroupWithDetails[]> => {
      // 获取所有 feature groups
      const groups = await featureApi.getFeatureGroups();

      // 并行请求每个 group 的 details
      const detailsPromises = groups.map((group) => featureApi.getFeatureGroupDetails(group.id));
      const details = await Promise.all(detailsPromises);

      // 聚合数据：将 group 基本信息和 features 合并
      return groups.map((group, index) => ({
        ...group,
        features: details[index].features,
      }));
    },
  });
};
