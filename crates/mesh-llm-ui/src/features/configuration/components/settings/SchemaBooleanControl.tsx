import { SchemaChoiceControl } from '@/features/configuration/components/settings/SchemaChoiceControl'
import type { SchemaSettingControlProps } from '@/features/configuration/components/settings/schema-control-utils'

export function SchemaBooleanControl(props: SchemaSettingControlProps) {
  return <SchemaChoiceControl {...props} />
}
