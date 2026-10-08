import { useQuery } from '@tanstack/react-query';
import { classifierKeys } from './query-keys';
import { classifierApi } from '@/services/api/classifier';

export function useClassifiers() {
  return useQuery({
    queryKey: classifierKeys.list(),
    queryFn: classifierApi.getClassifierList,
    select: (data) => {
      return data;
    },
  });
}
