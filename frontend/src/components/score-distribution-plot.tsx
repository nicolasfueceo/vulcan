'use client'

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { EvolutionIndividual } from '@/types/vulcan'

interface ScoreDistributionPlotProps {
  data: EvolutionIndividual[]
}

// Helper to create histogram data
const createHistogramData = (scores: number[], binCount: number = 10) => {
  if (scores.length === 0) return []

  const minScore = Math.min(...scores)
  const maxScore = Math.max(...scores)
  const binSize = (maxScore - minScore) / binCount
  
  if (binSize === 0) {
      return [{ range: minScore.toFixed(4), count: scores.length }]
  }

  const bins = Array.from({ length: binCount }, () => 0)

  for (const score of scores) {
    let binIndex = Math.floor((score - minScore) / binSize)
    if (binIndex === binCount) binIndex-- // Put max score in last bin
    bins[binIndex]++
  }

  return bins.map((count, i) => ({
    range: `${(minScore + i * binSize).toFixed(3)} - ${(minScore + (i + 1) * binSize).toFixed(3)}`,
    count,
  }))
}

export function ScoreDistributionPlot({ data }: ScoreDistributionPlotProps) {
  if (!data || data.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Score Distribution</CardTitle>
          <CardDescription>No population data available to display plot.</CardDescription>
        </CardHeader>
        <CardContent className="h-64 flex items-center justify-center">
          <p className="text-slate-500">Awaiting data...</p>
        </CardContent>
      </Card>
    )
  }

  const scores = data.map(ind => ind.score)
  const histogramData = createHistogramData(scores)

  return (
    <Card>
      <CardHeader>
        <CardTitle>Population Score Distribution</CardTitle>
        <CardDescription>Histogram of feature scores in the final population.</CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={histogramData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="range" angle={-30} textAnchor="end" height={70} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Legend />
            <Bar dataKey="count" fill="#8884d8" name="Number of Features" />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
} 