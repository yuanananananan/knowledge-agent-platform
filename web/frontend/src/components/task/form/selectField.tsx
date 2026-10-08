import {
  Field,
  FieldContent,
  FieldDescription,
  FieldLabel,
} from '@/components/ui/field';
import { useFieldContext } from '@/hooks/task/form';
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from '@/components/ui/select';

type SelectFieldProps = {
  title: string;
  description: string;
  items: {
    name: string;
    id: number;
    backboneType?: string;
  }[];
};

export default function SelectField({
  title,
  description,
  items,
}: SelectFieldProps) {
  const field = useFieldContext<number>();

  return (
    <Field orientation="vertical" className="lg:w-2/3 xl:w-1/2">
      <FieldContent>
        <FieldLabel>{title}</FieldLabel>
        <FieldDescription>{description}</FieldDescription>
      </FieldContent>

      <Select
        name={field.name}
        value={field.state.value ? String(field.state.value) : ''}
        onValueChange={(e) => {
          field.handleChange(Number(e));
        }}
      >
        <SelectTrigger>
          <SelectValue placeholder="请选择" />
        </SelectTrigger>

        <SelectContent position="item-aligned">
          {items.map((item) => (
            <SelectItem key={item.id} value={String(item.id)}>
              {item.name}
              {item.backboneType === 'nas_search' ? ' / NAS搜索主干' : ''}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </Field>
  );
}