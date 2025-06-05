'use client'

import { useState, useEffect, useCallback } from 'react'
import { VisualizationData } from '@/types/vulcan'

interface ExperimentListItem {
  id: number
  experiment_name: string
  algorithm: string
  start_time: string
  status: string
  iterations_completed: number
  best_score: number
  end_time?: string
}

interface UseExperimentDataResult {
  data: VisualizationData | null
  experiments: ExperimentListItem[]
  selectedExperiment: string | null
  setSelectedExperiment: (experimentName: string | null) => void
  isLoading: boolean
  error: string | null
  refreshData: () => void
  selectExperiment: (experimentName: string | null) => void
  isPolling: boolean
  setIsPolling: (polling: boolean) => void
}

export function useExperimentData(): UseExperimentDataResult {
  const [data, setData] = useState<VisualizationData | null>(null)
  const [experiments, setExperiments] = useState<ExperimentListItem[]>([])
  const [selectedExperiment, setSelectedExperiment] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isPolling, setIsPolling] = useState(false)

  const fetchExperiments = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8000/api/experiments')
      if (!response.ok) throw new Error('Failed to fetch experiments')
      
      const experimentsData = await response.json()
      setExperiments(experimentsData)
      
      return experimentsData
    } catch (err) {
      console.error('Error fetching experiments:', err)
      setError(err instanceof Error ? err.message : 'Unknown error')
      return []
    }
  }, [])

  const fetchExperimentData = useCallback(async (experimentName?: string) => {
    try {
      setIsLoading(true)
      setError(null)

      let url: string
      if (experimentName) {
        url = `http://localhost:8000/api/experiments/${experimentName}/data`
      } else {
        url = 'http://localhost:8000/api/experiments/latest/data'
      }

      const response = await fetch(url)
      if (!response.ok) throw new Error('Failed to fetch experiment data')
      
      const result = await response.json()
      
      if (result.status === 'success') {
        setData(result.data)
      } else {
        throw new Error(result.message || 'Failed to load experiment data')
      }
    } catch (err) {
      console.error('Error fetching experiment data:', err)
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setIsLoading(false)
    }
  }, [])

  const refreshData = useCallback(() => {
    fetchExperiments()
    fetchExperimentData(selectedExperiment || undefined)
  }, [fetchExperiments, fetchExperimentData, selectedExperiment])

  const selectExperiment = useCallback((experimentName: string | null) => {
    setSelectedExperiment(experimentName)
    if (experimentName) {
      fetchExperimentData(experimentName)
    } else {
      fetchExperimentData() // Fetch latest
    }
  }, [fetchExperimentData])

  // Initial data fetch
  useEffect(() => {
    refreshData()
  }, [])

  // Polling effect
  useEffect(() => {
    if (!isPolling) return

    const pollInterval = setInterval(() => {
      refreshData()
    }, 2000) // Poll every 2 seconds

    return () => clearInterval(pollInterval)
  }, [isPolling, refreshData])

  // Auto-select latest experiment when experiments list changes
  useEffect(() => {
    if (experiments.length > 0 && !selectedExperiment) {
      const latestExperiment = experiments[0] // Experiments are sorted by start time (newest first)
      if (latestExperiment.status === 'running') {
        setSelectedExperiment(latestExperiment.experiment_name)
        setIsPolling(true) // Auto-start polling for running experiments
      }
    }
  }, [experiments, selectedExperiment])

  // Stop polling when experiment finishes
  useEffect(() => {
    if (selectedExperiment && experiments.length > 0) {
      const currentExp = experiments.find(exp => exp.experiment_name === selectedExperiment)
      if (currentExp && currentExp.status === 'completed') {
        setIsPolling(false)
      }
    }
  }, [experiments, selectedExperiment])

  return {
    data,
    experiments,
    selectedExperiment,
    setSelectedExperiment,
    isLoading,
    error,
    refreshData,
    selectExperiment,
    isPolling,
    setIsPolling
  }
} 