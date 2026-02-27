import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { GlossaryText } from '../../src/components/game/GlossaryText.tsx';

describe('GlossaryText', () => {
  it('renders plain text unchanged', () => {
    render(<GlossaryText text="Hello world, no markers here." />);
    expect(screen.getByText('Hello world, no markers here.')).toBeInTheDocument();
  });

  it('renders glossary term with default display and tooltip', () => {
    render(<GlossaryText text="Enable {{fsdp}} for sharding." />);
    // "FSDP" is the glossary default display for "fsdp"
    expect(screen.getByText('FSDP')).toBeInTheDocument();
    expect(screen.getByText(/Enable/)).toBeInTheDocument();
    expect(screen.getByText(/for sharding\./)).toBeInTheDocument();
    // The tooltip wrapper should have cursor-help styling
    const term = screen.getByText('FSDP');
    expect(term).toHaveClass('cursor-help');
  });

  it('renders glossary term with custom display text', () => {
    render(<GlossaryText text="Use {{fsdp|Fully Sharded Data Parallel}} here." />);
    expect(screen.getByText('Fully Sharded Data Parallel')).toBeInTheDocument();
    const term = screen.getByText('Fully Sharded Data Parallel');
    expect(term).toHaveClass('cursor-help');
  });

  it('renders [text](url) as a clickable link', () => {
    render(
      <GlossaryText text="See [Flash Attention (Dao, 2022)](https://arxiv.org/abs/2205.14135) for details." />,
    );
    const link = screen.getByRole('link', { name: 'Flash Attention (Dao, 2022)' });
    expect(link).toHaveAttribute('href', 'https://arxiv.org/abs/2205.14135');
    expect(link).toHaveAttribute('target', '_blank');
    expect(link).toHaveAttribute('rel', 'noopener noreferrer');
  });

  it('renders mixed text, glossary terms, and links correctly', () => {
    render(
      <GlossaryText text="Train with {{tp}} and see [paper](https://example.com) for more." />,
    );
    expect(screen.getByText('TP')).toBeInTheDocument();
    expect(screen.getByText('TP')).toHaveClass('cursor-help');
    const link = screen.getByRole('link', { name: 'paper' });
    expect(link).toHaveAttribute('href', 'https://example.com');
    expect(screen.getByText(/Train with/)).toBeInTheDocument();
    expect(screen.getByText(/for more\./)).toBeInTheDocument();
  });

  it('renders unknown glossary term as plain text with dev warning', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    render(<GlossaryText text="Use {{nonexistent}} here." />);
    // Should render the term ID as plain text
    expect(screen.getByText(/nonexistent/)).toBeInTheDocument();
    // Should have logged a warning in dev mode
    expect(warnSpy).toHaveBeenCalledWith(
      '[GlossaryText] Unknown glossary term: "nonexistent"',
    );
    warnSpy.mockRestore();
  });

  it('renders adjacent glossary terms correctly', () => {
    render(<GlossaryText text="{{tp}}{{pp}}" />);
    expect(screen.getByText('TP')).toBeInTheDocument();
    expect(screen.getByText('PP')).toBeInTheDocument();
  });

  it('renders glossary term inside surrounding text', () => {
    render(<GlossaryText text="pre-{{fsdp}}-post" />);
    expect(screen.getByText('FSDP')).toBeInTheDocument();
    expect(screen.getByText(/pre-/)).toBeInTheDocument();
    expect(screen.getByText(/-post/)).toBeInTheDocument();
  });

  it('renders nothing for empty text', () => {
    const { container } = render(<GlossaryText text="" />);
    expect(container.innerHTML).toBe('');
  });

  it('renders multiple links in the same string', () => {
    render(
      <GlossaryText text="[link1](https://a.com) and [link2](https://b.com)" />,
    );
    const link1 = screen.getByRole('link', { name: 'link1' });
    const link2 = screen.getByRole('link', { name: 'link2' });
    expect(link1).toHaveAttribute('href', 'https://a.com');
    expect(link2).toHaveAttribute('href', 'https://b.com');
  });

  it('falls back to glossary default display when custom display is empty', () => {
    render(<GlossaryText text="Use {{fsdp|}} here." />);
    // Empty custom display should fall back to glossary default "FSDP"
    expect(screen.getByText('FSDP')).toBeInTheDocument();
  });

  it('renders backtick span as <code> element with correct styling', () => {
    render(<GlossaryText text="The formula is `GBS=64` here." />);
    const code = screen.getByText('GBS=64');
    expect(code.tagName).toBe('CODE');
    expect(code).toHaveClass('font-mono');
    expect(code).toHaveClass('bg-gray-800');
    expect(code).toHaveClass('rounded');
  });

  it('renders glossary, link, and backtick in the same string', () => {
    render(
      <GlossaryText text="Use {{fsdp}} with `TP=8` per [paper](https://example.com)." />,
    );
    // Glossary term
    const glossaryTerm = screen.getByText('FSDP');
    expect(glossaryTerm).toHaveClass('cursor-help');
    // Code span
    const code = screen.getByText('TP=8');
    expect(code.tagName).toBe('CODE');
    expect(code).toHaveClass('font-mono');
    // Link
    const link = screen.getByRole('link', { name: 'paper' });
    expect(link).toHaveAttribute('href', 'https://example.com');
    // Surrounding text
    expect(screen.getByText(/Use/)).toBeInTheDocument();
    expect(screen.getByText(/with/)).toBeInTheDocument();
  });

  it('renders multiple backtick spans', () => {
    render(<GlossaryText text="`GA = GBS/(MBS×DP)` gives `GA=4`" />);
    const codes = document.querySelectorAll('code');
    expect(codes).toHaveLength(2);
    expect(codes[0].textContent).toBe('GA = GBS/(MBS×DP)');
    expect(codes[1].textContent).toBe('GA=4');
  });

  it('does not render <code> when there are no backticks', () => {
    render(<GlossaryText text="no backtick code" />);
    const codes = document.querySelectorAll('code');
    expect(codes).toHaveLength(0);
  });
});
