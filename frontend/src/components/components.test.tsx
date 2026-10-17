import { describe, expect, it, vi } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ChannelSignals } from './search/ChannelSignals'
import { CitationList } from './search/CitationList'
import { ImportanceBreakdown } from './memory/ImportanceBreakdown'
import { AskPanel } from '@/pages/Search'
import { answer, hit, record } from '@/test/fixtures'

const wrap = (ui: React.ReactNode) =>
  render(<QueryClientProvider client={new QueryClient()}><MemoryRouter>{ui}</MemoryRouter></QueryClientProvider>)

describe('ChannelSignals', () => {
  it('shows the rank from each channel that found the memory and marks the others', () => {
    wrap(<ChannelSignals hit={hit} />)
    const box = screen.getByTestId('channel-signals')
    expect(within(box).getByText('keyword #1')).toBeInTheDocument()
    expect(within(box).getByText('semantic #1')).toBeInTheDocument()
    expect(within(box).getByText('graph —')).toBeInTheDocument()
  })
})

describe('CitationList', () => {
  it('links each statement to its source memory', () => {
    wrap(<CitationList result={answer} />)
    const link = screen.getByRole('link', { name: /source 032104fa/ })
    expect(link).toHaveAttribute('href', '/memories/032104fa-4d72-4ca5-bf5f-c71b2b480d33')
    expect(screen.getByText(/“I moved to Bengaluru.”/)).toBeInTheDocument()
  })
  it('shows the abstain reason instead of statements', () => {
    wrap(<CitationList result={{ ...answer, abstain: true, statements: [], reason: 'No memory names a place.' }} />)
    expect(screen.getByTestId('abstained')).toHaveTextContent('No memory names a place.')
  })
})

describe('ImportanceBreakdown', () => {
  it('shows the stored score, both parts and the model class', () => {
    wrap(<ImportanceBreakdown record={record()} />)
    const box = screen.getByTestId('importance')
    expect(box).toHaveTextContent('0.6846')
    expect(box).toHaveTextContent('0.8372')
    expect(box).toHaveTextContent('0.6337')
    expect(box).not.toHaveTextContent('do not add up')
  })
})

describe('AskPanel', () => {
  it('asks for confirmation before using Gemini and stays local when declined', async () => {
    const user = userEvent.setup()
    const fetch = vi.spyOn(globalThis, 'fetch')
    wrap(<AskPanel />)
    const toggle = screen.getByRole('switch', { name: 'Use Gemini' })
    await user.click(toggle)
    expect(await screen.findByText('Send memories to Gemini?')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Keep local' }))
    expect(toggle).not.toBeChecked()
    expect(screen.getByText(/nothing leaves this machine/)).toBeInTheDocument()
    await user.click(toggle)
    await user.click(await screen.findByRole('button', { name: 'Use Gemini' }))
    expect(toggle).toBeChecked()
    expect(fetch).not.toHaveBeenCalled()
    fetch.mockRestore()
  })
})
