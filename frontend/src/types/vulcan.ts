// Evolution Types
export interface EvolutionIndividual {
  id: string
  score: number
  generation: number
  parent_id?: string
  mutation_type?: string
  execution_successful: boolean
  repair_attempts: number
}

export interface EvolutionGenerationHistory {
  generation: number
  total_features: number
  successful_features: number
  avg_score: number
  best_score: number
  action_taken: string
  population_size: number
}

export interface AgentStats {
  [agentName: string]: {
    count: number
    reward_sum: number
  }
}

export interface EvolutionData {
  population: EvolutionIndividual[]
  generation_history: EvolutionGenerationHistory[]
  agent_stats: AgentStats
  best_candidate?: {
    feature_name: string
    score: number
    generation: number
  }
}

// LLM Interaction Types
export interface LLMFeatureOutput {
  name: string
  description: string
  feature_type: string
  code?: string
  llm_prompt?: string
  chain_of_thought_reasoning: string
}

export interface LLMInteractionLog {
  agent_name: string
  timestamp: number
  prompt_input: Record<string, any>
  raw_response: string
  parsed_response?: LLMFeatureOutput
  error_message?: string
}

// Decision Log Types
export interface DecisionLog {
  id: string
  timestamp: string
  iteration: number
  action: 'explore' | 'exploit'
  node_id: string
  exploration_threshold: number
  random_value: number
  ucb_values: Record<string, number>
  selected_node: string
  rationale: string
  feature_generated?: string
  score_achieved?: number
}

// Unified data type for components
export type VisualizationData = EvolutionData 