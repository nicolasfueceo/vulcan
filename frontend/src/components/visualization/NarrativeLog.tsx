'use client'

import React from 'react'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'
import { DecisionLog, LLMInteraction } from '@/types/vulcan'
import { Clock, Brain, Search, Lightbulb, Code } from 'lucide-react'

interface NarrativeLogProps {
  decisionLogs: DecisionLog[]
  llmInteractions: LLMInteraction[]
}

export function NarrativeLog({ decisionLogs, llmInteractions }: NarrativeLogProps) {
  // Combine and sort all events by timestamp
  const events = [
    ...decisionLogs.map(log => ({ type: 'decision', data: log, timestamp: log.timestamp })),
    ...llmInteractions.map(interaction => ({ type: 'llm', data: interaction, timestamp: interaction.timestamp }))
  ].sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString()
  }

  const getActionBadge = (action: string) => {
    const variants = {
      'explore': 'default',
      'exploit': 'secondary',
      'feature_generation': 'default',
      'feature_mutation': 'secondary',
      'reflection': 'outline'
    } as const

    const colors = {
      'explore': 'bg-blue-100 text-blue-800',
      'exploit': 'bg-orange-100 text-orange-800',
      'feature_generation': 'bg-green-100 text-green-800',
      'feature_mutation': 'bg-purple-100 text-purple-800',
      'reflection': 'bg-gray-100 text-gray-800'
    }

    return (
      <Badge 
        variant={variants[action as keyof typeof variants] || 'outline'}
        className={`${colors[action as keyof typeof colors] || ''} text-xs`}
      >
        {action.replace('_', ' ').toUpperCase()}
      </Badge>
    )
  }

  const renderDecisionEvent = (log: DecisionLog) => (
    <Card key={log.id} className="border-l-4 border-l-blue-500">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Search className="h-4 w-4 text-blue-600" />
            <CardTitle className="text-sm">MCTS Decision #{log.iteration}</CardTitle>
          </div>
          <div className="flex items-center gap-2">
            {getActionBadge(log.action)}
            <span className="text-xs text-slate-500">{formatTime(log.timestamp)}</span>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Decision Details */}
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="font-medium">Threshold:</span> {log.exploration_threshold.toFixed(3)}
          </div>
          <div>
            <span className="font-medium">Random Value:</span> {log.random_value.toFixed(3)}
          </div>
        </div>

        {/* UCB Values */}
        {Object.keys(log.ucb_values).length > 0 && (
          <div>
            <div className="text-sm font-medium mb-2">UCB Values:</div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              {Object.entries(log.ucb_values).map(([nodeId, value]) => (
                <div key={nodeId} className="flex justify-between p-2 bg-slate-50 rounded">
                  <span className="font-mono">{nodeId.slice(0, 8)}...</span>
                  <span className="font-bold">{value.toFixed(3)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Rationale */}
        <div>
          <div className="text-sm font-medium mb-1">Decision Rationale:</div>
          <div className="text-sm text-slate-700 bg-slate-50 p-2 rounded italic">
            "{log.rationale}"
          </div>
        </div>

        {/* Results */}
        {(log.feature_generated || log.score_achieved !== undefined) && (
          <div className="grid grid-cols-1 gap-2">
            {log.feature_generated && (
              <div>
                <span className="text-sm font-medium">Feature Generated:</span>
                <div className="font-mono text-xs bg-green-50 p-2 rounded mt-1">
                  {log.feature_generated}
                </div>
              </div>
            )}
            {log.score_achieved !== undefined && (
              <div>
                <span className="text-sm font-medium">Score Achieved:</span>
                <span className="ml-2 font-bold text-green-600">{log.score_achieved.toFixed(4)}</span>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )

  const renderLLMEvent = (interaction: LLMInteraction) => (
    <Card key={interaction.id} className="border-l-4 border-l-purple-500">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Brain className="h-4 w-4 text-purple-600" />
            <CardTitle className="text-sm">LLM Interaction</CardTitle>
          </div>
          <div className="flex items-center gap-2">
            {getActionBadge(interaction.action_type)}
            <span className="text-xs text-slate-500">{formatTime(interaction.timestamp)}</span>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Model Info */}
        <div className="flex items-center gap-4 text-sm">
          <div>
            <span className="font-medium">Model:</span> {interaction.model}
          </div>
          {interaction.node_id && (
            <div>
              <span className="font-medium">Node:</span> 
              <span className="font-mono ml-1">{interaction.node_id.slice(0, 8)}...</span>
            </div>
          )}
        </div>

        {/* Prompt */}
        <div>
          <div className="text-sm font-medium mb-1 flex items-center gap-1">
            <Code className="h-3 w-3" />
            Prompt:
          </div>
          <ScrollArea className="h-24 w-full border rounded p-2 bg-slate-50">
            <pre className="text-xs whitespace-pre-wrap">{interaction.prompt}</pre>
          </ScrollArea>
        </div>

        {/* Response */}
        <div>
          <div className="text-sm font-medium mb-1 flex items-center gap-1">
            <Lightbulb className="h-3 w-3" />
            Response:
          </div>
          <ScrollArea className="h-32 w-full border rounded p-2 bg-green-50">
            <pre className="text-xs whitespace-pre-wrap">{interaction.response}</pre>
          </ScrollArea>
        </div>
      </CardContent>
    </Card>
  )

  if (events.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-500">
        <div className="text-center">
          <div className="text-2xl mb-2">📋</div>
          <p>No decision logs available</p>
          <p className="text-sm">Decision details will appear here during MCTS execution</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Summary Stats */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-5 w-5" />
            Decision Log Summary
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-4 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-blue-600">{decisionLogs.length}</div>
              <div className="text-sm text-slate-600">Decisions</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-purple-600">{llmInteractions.length}</div>
              <div className="text-sm text-slate-600">LLM Calls</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-green-600">
                {decisionLogs.filter(log => log.action === 'explore').length}
              </div>
              <div className="text-sm text-slate-600">Explorations</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-orange-600">
                {decisionLogs.filter(log => log.action === 'exploit').length}
              </div>
              <div className="text-sm text-slate-600">Exploitations</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Event Timeline */}
      <div className="space-y-3 max-h-[600px] overflow-y-auto">
        {events.map((event) => (
          <div key={`${event.type}-${event.data.id}`}>
            {event.type === 'decision' ? 
              renderDecisionEvent(event.data as DecisionLog) : 
              renderLLMEvent(event.data as LLMInteraction)
            }
          </div>
        ))}
      </div>
    </div>
  )
} 