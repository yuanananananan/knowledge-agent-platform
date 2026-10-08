import { useQuery } from '@tanstack/react-query';
import { analysisApi } from '@/services/api/analysis';
import { analysisKeys } from './query-keys';

export function useAnalysisRecords(userId: number, page = 1, pageSize = 10) {
  return useQuery({
    queryKey: [...analysisKeys.lists(userId), page, pageSize],
    queryFn: () => analysisApi.getAnalysisList(userId, page, pageSize),
    enabled: !!userId,
  });
}