import type { ColumnDef } from '@tanstack/react-table';
import { Checkbox } from '@/components/ui/checkbox';
import type { Feature } from '@/types/feature';
import type { Modality } from '@/types/dataset';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { getModalityLabel } from '@/lib/modality';

export const createColumns = (): ColumnDef<Feature>[] => [
  {
    id: 'select',
    header: ({ table }) => {
      return (
        <Checkbox
          checked={
            table.getIsAllPageRowsSelected() ||
            (table.getIsSomePageRowsSelected() && 'indeterminate')
          }
          onCheckedChange={(value) => table.toggleAllPageRowsSelected(!!value)}
          aria-label="Select all"
        />
      );
    },
    cell: ({ row }) => {
      return (
        <Checkbox
          checked={row.getIsSelected()}
          onCheckedChange={(value) => row.toggleSelected(!!value)}
          aria-label="Select row"
        />
      );
    },
    enableSorting: false,
    enableHiding: false,
    size: 50,
  },
  {
    accessorKey: 'id',
    header: 'ID',
    cell: ({ row }) => <div className="w-[60px]">{row.getValue('id')}</div>,
    size: 80,
  },
  {
    accessorKey: 'name',
    header: '算子名称',
    cell: ({ row }) => {
      const name = row.getValue('name') as string;
      return <div className="font-medium">{name}</div>;
    },
    size: 150,
  },
  {
    accessorKey: 'modality',
    header: '模态',
    cell: ({ row }) => {
      const modality = row.getValue('modality') as Modality;
      return <div className="font-medium w-20">{getModalityLabel(modality)}</div>;
    },
    size: 100,
  },
  {
    accessorKey: 'description',
    header: '描述',
    cell: ({ row }) => {
      const description = row.getValue('description') as string;
      return (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="text-sm text-muted-foreground truncate cursor-help">
                {description}
              </div>
            </TooltipTrigger>
            <TooltipContent side="top" className="max-w-md">
              <p>{description}</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      );
    },
    size: 200,
  },
];
