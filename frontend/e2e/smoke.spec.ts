import { expect, test } from '@playwright/test'

test('sign up, remember a move, see history, ask, graph, system, delete', async ({ page }) => {
  const email = `e2e-${Date.now()}@example.com`
  await page.goto('/app/')
  await expect(page).toHaveURL(/\/app\/login$/)

  await page.getByRole('tab', { name: 'Sign up' }).click()
  // By id: while the tab switch animates, the sign-in form is still mounted and also has fields labelled Email/Password.
  await page.locator('#signup-email').fill(email)
  await page.locator('#signup-password').fill('a-very-strong-password')
  await page.getByRole('button', { name: 'Create account' }).click()
  await expect(page.getByRole('heading', { name: 'Tell CogniMem something' })).toBeVisible()

  // Chat: each stored message shows how it was classified and scored.
  for (const text of ['I live in Chennai.', 'I moved to Bengaluru.']) {
    await page.getByLabel('Message').fill(text)
    await page.getByRole('button', { name: 'Remember' }).click()
    await expect(page.getByRole('link', { name: text })).toBeVisible()
  }
  await expect(page.getByTestId('importance').first()).toContainText('Rule-based score only')

  // Memories: the move superseded the old city; history shows it.
  await page.getByRole('link', { name: 'Memories' }).click()
  await expect(page.getByRole('link', { name: 'I moved to Bengaluru.' })).toBeVisible()
  await expect(page.getByRole('link', { name: 'I live in Chennai.' })).toHaveCount(0)
  await page.getByRole('switch', { name: 'Include history' }).click()
  await expect(page.getByRole('link', { name: 'I live in Chennai.' }).first()).toBeVisible()
  await expect(page.getByText('superseded').first()).toBeVisible()

  // Memory detail: revision timeline from the history endpoint, and a reload of the deep link works.
  await page.getByRole('link', { name: 'I live in Chennai.' }).first().click()
  await expect(page.getByText('Revision history')).toBeVisible()
  await page.reload()
  await expect(page.getByText('Revision history')).toBeVisible()

  // Ask: answered locally, citing the current city only.
  await page.getByRole('link', { name: 'Search & Ask' }).click()
  await page.getByLabel('Question').fill('Where do I live now?')
  await page.getByLabel('Retrieval').selectOption('keyword')
  await page.getByRole('button', { name: 'Ask', exact: true }).click()
  await expect(page.getByText('“I moved to Bengaluru.”', { exact: false })).toBeVisible()
  await expect(page.getByText('“I live in Chennai.”', { exact: false })).toHaveCount(0)

  // Graph: the residence relation, in the table view.
  await page.getByRole('link', { name: 'Graph' }).click()
  await page.getByRole('switch', { name: 'Table view' }).click()
  await expect(page.getByRole('cell', { name: 'Bengaluru' })).toBeVisible()
  await expect(page.getByRole('cell', { name: 'Chennai' })).toHaveCount(0)

  // Evaluation: the README's numbers from the tracked snapshots, the model report, and a snapshot comparison.
  await page.getByRole('link', { name: 'Evaluation' }).click()
  const retention = page.getByRole('row', { name: /LongMemEval user-fact retention/ })
  await expect(retention).toContainText('21.0%')
  await expect(retention).toContainText('90.6%')
  await expect(page.getByRole('row', { name: /Hybrid \+ reranker \(shipped\)/ })).toContainText('97.5%')
  await page.getByRole('tab', { name: 'Importance model' }).click()
  await expect(page.getByRole('row', { name: /Ensemble \+ thresholds \(shipped\)/ })).toContainText('0.8912')
  await page.getByRole('tab', { name: 'Compare snapshots' }).click()
  await page.getByLabel('Filter').fill('longmemeval all memory_retention')
  await expect(page.getByRole('row', { name: /longmemeval\.all\.memory_retention/ })).toContainText('+0.6952')

  // System: ready, heuristic scorer; then delete the account behind a typed confirmation.
  await page.getByRole('link', { name: 'System' }).click()
  await expect(page.getByText('memory store: ok')).toBeVisible()
  await expect(page.getByText('Rules only (heuristic)')).toBeVisible()
  await page.getByRole('button', { name: 'Delete account and all memories' }).click()
  const confirm = page.getByRole('button', { name: 'Delete permanently' })
  await expect(confirm).toBeDisabled()
  await page.getByPlaceholder(email).fill(email)
  await confirm.click()
  await expect(page.getByText('Your account was deleted.')).toBeVisible()
})
