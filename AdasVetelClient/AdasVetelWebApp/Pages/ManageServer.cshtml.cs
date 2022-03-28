using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;

namespace AdasVetelWebApp.Pages;

public class ManageServerModel : PageModel
{
    private readonly ILogger<ManageServerModel> _logger;

    public ManageServerModel(ILogger<ManageServerModel> logger)
    {
        _logger = logger;
    }

    public void OnGet()
    {
    }
}

