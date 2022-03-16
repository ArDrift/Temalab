using AdasVetelServer.model;
using System.Collections.Generic;
using System.Text.Json;

namespace AdasVetelServer.db
{
    public delegate List<T> NoIdCallBack<T>();
    class NoIdRouteMapper<T> : RouteMapperBase where T : DbElement
    {
        private event NoIdCallBack<T> CallBack;
        public NoIdRouteMapper(string route, NoIdCallBack<T> callback) : base(route)
        {
            CallBack = callback;
        }

        public override List<byte[]> RunQuery(string route)
        {
            List<byte[]> res = new List<byte[]>();
            foreach (T item in CallBack())
                res.Add(JsonSerializer.SerializeToUtf8Bytes(item, new JsonSerializerOptions { WriteIndented = true }));
            return res;
        }
    }
}
