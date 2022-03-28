using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;

namespace AdasVetelWebApp.Pages;

public class ManageContractsModel : PageModel
{
    private readonly ILogger<ManageContractsModel> _logger;

    public ManageContractsModel(ILogger<ManageContractsModel> logger)
    {
        _logger = logger;
    }

    public void OnGet()
    {
    }
}

