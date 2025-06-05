'use client'

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'

interface ActionPerformancePlotProps {
  data: {
    generate_new: number[]
    mutate_existing: number[]
  }
}

// Helper function to calculate statistics for a box plot
const getStats = (data: number[]) => {
  if (!data || data.length === 0) {
    return { min: 0, q1: 0, median: 0, q3: 0, max: 0 }
  }
  const sorted = [...data].sort((a, b) => a - b)
  const q1 = sorted[Math.floor(sorted.length / 4)]
  const median = sorted[Math.floor(sorted.length / 2)]
  const q3 = sorted[Math.floor((sorted.length * 3) / 4)]
  return {
    min: sorted[0],
    q1,
    median,
    q3,
    max: sorted[sorted.length - 1],
  }
}

export function ActionPerformancePlot({ data }: ActionPerformancePlotProps) {
  if (!data || (!data.generate_new?.length && !data.mutate_existing?.length)) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Operator Performance</CardTitle>
          <CardDescription>No action reward data available to display plot.</CardDescription>
        </CardHeader>
        <CardContent className="h-64 flex items-center justify-center">
          <p className="text-slate-500">Awaiting data...</p>
        </CardContent>
      </Card>
    )
  }

  const generateStats = getStats(data.generate_new || [])
  const mutateStats = getStats(data.mutate_existing || [])

  const chartData = [
    { name: 'Generate New', ...generateStats },
    { name: 'Mutate Existing', ...mutateStats },
  ]
  
  // A simplified box plot using Bar components
  return (
    <Card>
      <CardHeader>
        <CardTitle>Operator Performance</CardTitle>
        <CardDescription>Comparing scores from different feature creation methods.</CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="median" fill="#8884d8" name="Median Score" />
            <Bar dataKey="q1" fill="#82ca9d" name="1st Quartile" />
            <Bar dataKey="q3" fill="#ffc658" name="3rd Quartile" />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
} 