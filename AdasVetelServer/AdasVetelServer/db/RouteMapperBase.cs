using System.Collections.Generic;

namespace AdasVetelServer.db
{
    abstract class RouteMapperBase
    {
        public string Route { get; set; }
        public RouteMapperBase(string route)
        {
            Route = route;
        }

        public abstract List<byte[]> RunQuery(string route);
    }
}
