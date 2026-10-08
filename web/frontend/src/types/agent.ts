export type AgentMessage = {
  role: 'user' | 'assistant';
  content: string;
};

export type AgentConversation = {
  conversation_id: string;
  title: string;
  checkpoint_version: number;
  created_at: string;
  updated_at: string;
};

export type AgentTraceEvent = {
  run_id: string;
  sequence: number;
  type:
    | 'run.accepted'
    | 'run.created'
    | 'run.collecting'
    | 'run.reasoning'
    | 'run.started'
    | 'model.delta'
    | 'tool.started'
    | 'tool.completed'
    | 'workflow.node.completed'
    | 'run.completed'
    | 'run.failed'
    | 'run.cancelled';
  payload: Record<string, unknown>;
};

export type AgentRunSummary = {
  run_id: string;
  conversation_id: string;
  status: 'pending' | 'collecting_evidence' | 'reasoning' | 'succeeded' | 'failed' | 'cancelled';
  answer?: string;
  error?: string;
};

export type ActiveRun = {
  runId: string;
  conversationId: string;
  lastEventId: number;
  status: AgentRunSummary['status'];
};

export type GoldenCase = {
  case_id: string;
  name: string;
  category: string;
  prompt: string;
  required_keywords: string[];
  forbidden_keywords: string[];
  expected_intent: string | null;
  expected_tools: string[];
  enabled: boolean;
};

export type EvalRun = {
  eval_run_id: string;
  status: string;
  total: number;
  passed: number;
  score: number;
  metrics: {
    task_success_rate?: number;
    tool_accuracy?: number;
    intent_accuracy?: number;
    average_latency_ms?: number;
    p95_latency_ms?: number;
    failure_rate?: number;
    failed_cases?: number;
    completed_cases?: number;
    progress?: number;
    error?: string;
  };
  created_at: string;
  completed_at: string | null;
};

export type AgentActionProposal = {
  proposal_id: string;
  conversation_id: string | null;
  action_type: string;
  payload: Record<string, unknown>;
  reason: string | null;
  impact: string;
  status: 'pending_approval' | 'executing' | 'succeeded' | 'failed' | 'rejected' | 'expired';
  result: Record<string, unknown> | null;
  error_message: string | null;
  expires_at: string;
  approved_at: string | null;
  executed_at: string | null;
  created_at: string;
  updated_at: string;
};

export type AgentActionVerification = {
  verified: boolean;
  target: string;
  [key: string]: unknown;
};

export type AgentActionDetail = {
  proposal: AgentActionProposal;
  verification: AgentActionVerification;
};
