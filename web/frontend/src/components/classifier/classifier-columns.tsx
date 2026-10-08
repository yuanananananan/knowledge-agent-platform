import type { ColumnDef } from '@tanstack/react-table';
import type { Classifier } from '@/types/classifier';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

export const columns: ColumnDef<Classifier>[] = [
  {
    accessorKey: 'id',
    header: 'ID',
    cell: ({ row }) => <div className="w-[60px]">{row.getValue('id')}</div>,
    size: 80,
  },
  {
    accessorKey: 'name',
    header: '名称',
    cell: ({ row }) => {
      const name = row.getValue('name') as string;
      return <div className="font-medium">{name}</div>;
    },
    size: 150,
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
    size: 400,
  },
];
