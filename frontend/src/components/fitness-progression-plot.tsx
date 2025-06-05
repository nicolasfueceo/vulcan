'use client'

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { EvolutionGenerationHistory } from '@/types/vulcan'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'

interface FitnessProgressionPlotProps {
  data: EvolutionGenerationHistory[]
}

export function FitnessProgressionPlot({ data }: FitnessProgressionPlotProps) {
  if (!data || data.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Fitness Progression</CardTitle>
          <CardDescription>No generation data available to display plot.</CardDescription>
        </CardHeader>
        <CardContent className="h-64 flex items-center justify-center">
          <p className="text-slate-500">Awaiting data...</p>
        </CardContent>
      </Card>
    )
  }

  const chartData = data.map(gen => ({
    name: `Gen ${gen.generation}`,
    'Best Score': gen.best_score,
    'Average Score': gen.avg_score,
  }))

  return (
    <Card>
      <CardHeader>
        <CardTitle>Fitness Progression Over Generations</CardTitle>
        <CardDescription>
          Tracking the best and average population scores over time.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart
            data={chartData}
            margin={{
              top: 5,
              right: 30,
              left: 20,
              bottom: 5,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis domain={['auto', 'auto']} />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(255, 255, 255, 0.8)',
                backdropFilter: 'blur(5px)',
                border: '1px solid #ccc',
                borderRadius: '8px',
              }}
            />
            <Legend />
            <Line type="monotone" dataKey="Best Score" stroke="#8884d8" strokeWidth={2} activeDot={{ r: 8 }} />
            <Line type="monotone" dataKey="Average Score" stroke="#82ca9d" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
} 