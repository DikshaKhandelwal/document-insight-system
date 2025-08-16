'use client'

import * as React from 'react'
import {
  ThemeProvider as NextThemesProvider,
  type ThemeProviderProps,
} from 'next-themes'

export function ThemeProvider({ children, ...props }: ThemeProviderProps) {
  const [mounted, setMounted] = React.useState(false)

  React.useEffect(() => {
    setMounted(true)
  }, [])

  // Always render children immediately, but without theme classes during SSR
  if (!mounted) {
    return <div suppressHydrationWarning>{children}</div>
  }

  return (
    <NextThemesProvider 
      {...props}
      attribute="class"
      defaultTheme="light"
      enableSystem={false}
      storageKey="document-insight-theme"
      disableTransitionOnChange
    >
      {children}
    </NextThemesProvider>
  )
}
