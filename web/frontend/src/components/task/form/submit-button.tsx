import { Button } from '@/components/ui/button';
import { useFormContext } from '@/hooks/task/form';

export default function SubmitButton() {
  const form = useFormContext();
  return (
    <form.Subscribe selector={(state) => state.isSubmitting}>
      {(isSubmitting) => (
        <Button type="submit" disabled={isSubmitting}>
          提交
        </Button>
      )}
    </form.Subscribe>
  );
}
