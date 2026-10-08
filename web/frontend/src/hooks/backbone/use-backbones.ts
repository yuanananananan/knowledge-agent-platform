import { useQuery } from '@tanstack/react-query';
import { backboneKeys } from './query-keys';
import { backboneApi } from '@/services/api/backbone';

export function useBackbones() {
  return useQuery({
    queryKey: backboneKeys.list(),
    queryFn: backboneApi.getBackboneList,
    select: (data) => {
      return data;
    },
  });
}
