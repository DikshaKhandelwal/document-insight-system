import { FileText, Settings, Info } from "lucide-react"
import { Button } from "@/components/ui/button"

export default function HeaderBar() {
  return (
    <header className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-6">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg adobe-gradient flex items-center justify-center">
          <FileText className="w-5 h-5 text-white" />
        </div>
        <h1 className="font-serif text-xl font-semibold text-slate-800">Document Insight</h1>
      </div>

      <div className="flex items-center gap-2">
        <Button variant="ghost" size="sm">
          <Settings className="w-4 h-4" />
        </Button>
        <Button variant="ghost" size="sm">
          <Info className="w-4 h-4" />
        </Button>
      </div>
    </header>
  )
}
