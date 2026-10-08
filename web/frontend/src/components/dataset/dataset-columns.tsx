import type { ColumnDef } from '@tanstack/react-table';
import { Checkbox } from '@/components/ui/checkbox';
import type { Dataset, Modality } from '@/types/dataset';
import { Button } from '@/components/ui/button';
import { getModalityLabel } from '@/lib/modality';

interface ColumnCallbacks {
  onNameClick: (dataset: Dataset) => void;
}

export const createColumns = (callbacks: ColumnCallbacks): ColumnDef<Dataset>[] => [
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
    header: '数据集',
    cell: ({ row }) => {
      const dataset = row.original;
      const name = row.getValue('name') as string;
      return (
        <Button
          className="p-0"
          variant="link"
          onClick={() => {
            callbacks.onNameClick(dataset);
          }}
        >
          {name}
        </Button>
      );
    },
    size: 100,
  },
  {
    accessorKey: 'modality',
    header: '模态',
    cell: ({ row }) => {
      const modality = row.getValue('modality') as Modality;
      return <div className="font-medium w-32">{getModalityLabel(modality)}</div>;
    },
    size: 100,
  },
  {
    accessorKey: 'path',
    header: '路径',
    cell: ({ row }) => {
      const path = row.getValue('path') as string;
      return <div className="font-medium w-32">{path}</div>;
    },
    size: 100,
  },
];
