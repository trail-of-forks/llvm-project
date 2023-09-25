
#include "llvm/MC/CCRegistry.h"
#include "llvm/ADT/iterator_range.h"
#include <map>
#include <memory>




namespace llvm {
    static CCRegistry::override_map OverrideMap;

    void CCRegistry::registerCCOverrride(const char* TargetName, std::unique_ptr<CCObj> Handler) {
        OverrideMap.emplace(TargetName, std::move(Handler));
    }

    CCRegistry::it_range CCRegistry::findCCOverrides(llvm::StringRef TargetName) {
        auto pr = OverrideMap.equal_range(std::string(TargetName));
        return CCRegistry::it_range(pr.first,pr.second); 
    }

} // namespace llvm