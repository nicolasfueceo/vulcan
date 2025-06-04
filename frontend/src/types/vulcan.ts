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

export interface EvolutionData {
  population: EvolutionIndividual[]
  generation_history: EvolutionGenerationHistory[]
  action_rewards: {
    generate_new: number[]
    mutate_existing: number[]
  }
  best_candidate?: {
    feature_name: string
    score: number
    generation: number
  }
}

// MCTS Types
export interface MCTSNode {
  id: string
  parent_id?: string
  depth: number
  visits: number
  value: number
  ucb_value: number
  is_terminal: boolean
  is_fully_expanded: boolean
  action_taken: 'explore' | 'exploit' | 'root'
  feature_name?: string
  feature_code?: string
  score?: number
  children: string[]
  cumulative_features: string[]
}

export interface MCTSEdge {
  id: string
  source: string
  target: string
  action_type: 'explore' | 'exploit'
}

export interface MCTSStats {
  total_nodes: number
  max_depth: number
  best_score: number
  iterations_completed: number
  avg_branching_factor: number
}

export interface MCTSData {
  nodes: MCTSNode[]
  edges: MCTSEdge[]
  best_node_id?: string
  stats: MCTSStats
  generation_history: EvolutionGenerationHistory[]  // For compatibility
  action_rewards: {
    generate_new: number[]
    mutate_existing: number[]
  }
  best_candidate?: {
    feature_name: string
    score: number
    generation: number
  }
}

// LLM Interaction Types
export interface LLMInteraction {
  id: string
  timestamp: string
  prompt: string
  response: string
  model: string
  action_type: 'feature_generation' | 'feature_mutation' | 'reflection'
  node_id?: string
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
export type VisualizationData = EvolutionData | MCTSData 